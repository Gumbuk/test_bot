import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import schedule
import threading
import os

# 텔레그램 설정
token = os.getenv('TELEGRAM_TOKEN')
chat_id = os.getenv('CHAT_ID')
url = f'https://api.telegram.org/bot{token}/sendMessage'

class TradingStrategyScanner:
    def __init__(self, interval='15m', volume_threshold=1000000):
        self.base_url = "https://api.binance.com/api/v3"
        self.interval = interval
        self.volume_threshold = volume_threshold
        
        # 시간대별 설정
        self.interval_map = {
            '1m': '1m',
            '3m': '3m', 
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        
    def send_telegram_message(self, message):
        """텔레그램 메시지 전송"""
        try:
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print(f"✅ 텔레그램 메시지 전송 성공")
            else:
                print(f"❌ 텔레그램 메시지 전송 실패: {response.status_code}")
        except Exception as e:
            print(f"❌ 텔레그램 메시지 전송 오류: {str(e)}")
    
    def get_high_volume_coins(self):
        """거래량이 높은 USDT 페어 코인들 조회"""
        url = f"{self.base_url}/ticker/24hr"
        response = requests.get(url)
        data = response.json()
        
        high_volume_coins = []
        for item in data:
            try:
                if item['symbol'].endswith('USDT'):
                    volume_usdt = float(item['quoteVolume'])
                    if volume_usdt >= self.volume_threshold:
                        high_volume_coins.append({
                            "symbol": item['symbol'],
                            "volume": volume_usdt,
                            "price": float(item['lastPrice'])
                        })
            except Exception:
                pass
        
        return high_volume_coins
    
    def get_klines_data(self, symbol, limit=500):
        """캔들스틱 데이터 조회"""
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 숫자형으로 변환
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def calculate_heikin_ashi(self, df):
        """하이킨 아시 캔들 계산"""
        ha_df = df.copy()
        
        # 하이킨 아시 공식
        ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        ha_df['HA_Open'] = 0.0
        ha_df.loc[0, 'HA_Open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        for i in range(1, len(ha_df)):
            ha_df.loc[i, 'HA_Open'] = (ha_df.loc[i-1, 'HA_Open'] + ha_df.loc[i-1, 'HA_Close']) / 2
        
        ha_df['HA_High'] = ha_df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        return ha_df
    
    def calculate_ema(self, df, period=200):
        """지수이동평균 계산 (talib 없이)"""
        alpha = 2 / (period + 1)
        ema = df['close'].copy()
        ema.iloc[0] = df['close'].iloc[0]
        
        for i in range(1, len(df)):
            ema.iloc[i] = alpha * df['close'].iloc[i] + (1 - alpha) * ema.iloc[i-1]
        
        return ema
    
    def calculate_rsi(self, df, period=14):
        """RSI 계산 (더 정확한 버전)"""
        delta = df['close'].diff()
        
        # 상승분과 하락분 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 평균 계산 (첫 번째 값은 단순평균, 이후는 지수평균)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 이후 지수평균으로 업데이트
        for i in range(period, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
        # RSI 계산
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_stochastic_rsi(self, df, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """스토캐스틱 RSI 계산 (정확한 버전)"""
        # RSI 계산
        rsi = self.calculate_rsi(df, rsi_period)
        
        # 스토캐스틱 RSI 계산
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        
        # 0으로 나누기 방지
        denominator = rsi_max - rsi_min
        stoch_rsi = np.where(denominator != 0, (rsi - rsi_min) / denominator, 0)
        
        # K, D 선 계산 (이동평균 대신 스무딩 적용)
        k_line = pd.Series(stoch_rsi).rolling(window=k_period).mean() * 100
        d_line = k_line.rolling(window=d_period).mean()
        
        return k_line, d_line
    
    def check_heikin_ashi_signals(self, ha_df):
        """하이킨 아시 신호 확인"""
        if len(ha_df) < 2:
            return None
        
        current = ha_df.iloc[-1]
        previous = ha_df.iloc[-2]
        
        # 현재 캔들 정보
        current_body = abs(current['HA_Close'] - current['HA_Open'])
        current_upper_shadow = current['HA_High'] - max(current['HA_Open'], current['HA_Close'])
        current_lower_shadow = min(current['HA_Open'], current['HA_Close']) - current['HA_Low']
        
        # 이전 캔들 정보
        previous_color = 'GREEN' if previous['HA_Close'] > previous['HA_Open'] else 'RED'
        previous_body = abs(previous['HA_Close'] - previous['HA_Open'])
        
        # 매수 신호: 발아래 꼬리가 없는 초록색 양봉이 이전 몸통보다 크면
        if (current['HA_Close'] > current['HA_Open'] and  # 초록색 양봉
            current_lower_shadow == 0 and  # 발아래 꼬리 없음
            previous_color == 'GREEN' and  # 이전 캔들이 초록색
            current_body > previous_body):  # 이전 몸통보다 큼
            return 'BUY'
        
        # 매도 신호: 윗꼬리가 없는 빨간색 음봉이 이전 몸통보다 크면
        elif (current['HA_Close'] < current['HA_Open'] and  # 빨간색 음봉
              current_upper_shadow == 0 and  # 윗꼬리 없음
              previous_color == 'RED' and  # 이전 캔들이 빨간색
              current_body > previous_body):  # 이전 몸통보다 큼
            return 'SELL'
        
        return None
    
    def check_ema_position(self, df, ema_200):
        """200 EMA 위치 확인"""
        if len(df) == 0 or pd.isna(ema_200.iloc[-1]):
            return None
        
        current_price = df['close'].iloc[-1]
        current_ema = ema_200.iloc[-1]
        
        if current_price > current_ema:
            return 'ABOVE_EMA'  # 상승 추세
        else:
            return 'BELOW_EMA'  # 하락 추세
    
    def check_stochastic_rsi_signals(self, k_line, d_line):
        """스토캐스틱 RSI 신호 확인 (더 정확한 버전)"""
        if len(k_line) < 2 or len(d_line) < 2:
            return None
        
        current_k = k_line.iloc[-1]
        current_d = d_line.iloc[-1]
        prev_k = k_line.iloc[-2]
        prev_d = d_line.iloc[-2]
        
        # NaN 값 체크
        if pd.isna(current_k) or pd.isna(current_d) or pd.isna(prev_k) or pd.isna(prev_d):
            return None
        
        # 과매도 구간에서 K선이 D선을 상향 돌파 (매수 신호)
        if (current_k < 20 and current_d < 20 and 
            prev_k <= prev_d and current_k > current_d):
            return 'BUY'
        
        # 과매수 구간에서 K선이 D선을 하향 돌파 (매도 신호)
        elif (current_k > 80 and current_d > 80 and 
              prev_k >= prev_d and current_k < current_d):
            return 'SELL'
        
        return None
    
    def scan_coin(self, coin_info):
        """개별 코인 스캔"""
        try:
            symbol = coin_info['symbol']
            
            # 캔들스틱 데이터 조회
            df = self.get_klines_data(symbol, limit=300)
            if len(df) < 200:
                return None
            
            # 하이킨 아시 계산
            ha_df = self.calculate_heikin_ashi(df)
            
            # 200 EMA 계산
            ema_200 = self.calculate_ema(df, period=200)
            
            # 스토캐스틱 RSI 계산
            k_line, d_line = self.calculate_stochastic_rsi(df)
            
            # 신호 확인
            ha_signal = self.check_heikin_ashi_signals(ha_df)
            ema_position = self.check_ema_position(df, ema_200)
            stoch_rsi_signal = self.check_stochastic_rsi_signals(k_line, d_line)
            
            # 전략 조건 확인 (수정된 조건)
            if ha_signal and ema_position and stoch_rsi_signal:
                # 롱 포지션: 200EMA 위 + 과매도에서 상향 돌파 + 하이킨 아시 매수 신호
                if (ha_signal == 'BUY' and 
                    ema_position == 'ABOVE_EMA' and 
                    stoch_rsi_signal == 'BUY'):
                    return {
                        'symbol': symbol,
                        'signal': 'LONG',
                        'price': coin_info['price'],
                        'volume': coin_info['volume'],
                        'ha_signal': ha_signal,
                        'ema_position': ema_position,
                        'stoch_rsi_signal': stoch_rsi_signal
                    }
                # 숏 포지션: 200EMA 아래 + 과매수에서 하향 돌파 + 하이킨 아시 매도 신호
                elif (ha_signal == 'SELL' and 
                      ema_position == 'BELOW_EMA' and 
                      stoch_rsi_signal == 'SELL'):
                    return {
                        'symbol': symbol,
                        'signal': 'SHORT',
                        'price': coin_info['price'],
                        'volume': coin_info['volume'],
                        'ha_signal': ha_signal,
                        'ema_position': ema_position,
                        'stoch_rsi_signal': stoch_rsi_signal
                    }
            
            return None
            
        except Exception as e:
            print(f"에러 발생 ({symbol}): {str(e)}")
            return None
    
    def run_scanner(self):
        """메인 스캐너 실행"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n=== 스캔 시작: {current_time} ===")
        print(f"시간대: {self.interval}")
        print(f"거래량 기준: {self.volume_threshold:,} USDT 이상")
        
        # 고거래량 코인 조회
        high_volume_coins = self.get_high_volume_coins()
        print(f"고거래량 코인 {len(high_volume_coins)}개 발견")
        
        # 각 코인 스캔
        signals = []
        for i, coin in enumerate(high_volume_coins, 1):
            signal = self.scan_coin(coin)
            if signal:
                signals.append(signal)
                print(f"✅ 신호 발견: {signal['symbol']} - {signal['signal']}")
            
            # API 호출 제한 방지
            time.sleep(0.1)
        
        # 텔레그램 메시지 생성
        if signals:
            message = self.create_telegram_message(signals, current_time)
            self.send_telegram_message(message)
        else:
            print(f"찾은 코인 없음")        

        print(f"스캔 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"발견된 신호: {len(signals)}개")
    
    def create_telegram_message(self, signals, scan_time):
        """텔레그램 메시지 생성"""
        message = f"🔍 <b>암호화폐 스캔 결과</b>\n"
        message += f"⏰ 스캔 시간: {scan_time}\n"
        message += f"📊 시간대: {self.interval}\n"
        message += f"💰 거래량 기준: {self.volume_threshold:,} USDT 이상\n\n"
        
        if signals:
            message += f"🎯 <b>발견된 신호: {len(signals)}개</b>\n\n"
            
            for i, signal in enumerate(signals, 1):
                emoji = "🟢" if signal['signal'] == 'LONG' else "🔴"
                message += f"{emoji} <b>{signal['signal']} 신호 #{i}</b>\n"
                message += f"📈 코인: {signal['symbol']}\n"
                message += f"💵 현재가: ${signal['price']:.6f}\n"
                message += f"📊 24h 거래량: {signal['volume']:,.0f} USDT\n"
                message += f"📋 하이킨 아시: {signal['ha_signal']}\n"
                message += f"📈 200 EMA: {signal['ema_position']}\n"
                message += f"📊 스토캐스틱 RSI: {signal['stoch_rsi_signal']}\n\n"
        else:
            message += "❌ <b>현재 조건을 만족하는 코인이 없습니다.</b>\n\n"
        
        message += f"⏰ 다음 스캔: 매 시각 10분, 25분, 40분, 55분"
        
        return message

def run_scheduled_scan():
    """스케줄된 스캔 실행"""
    scanner = TradingStrategyScanner(interval='15m', volume_threshold=1000000)
    scanner.run_scanner()

def main():
    parser = argparse.ArgumentParser(description='하이킨 아시 + 200EMA + 스토캐스틱 RSI 전략 스캐너 (스케줄러)')
    parser.add_argument('--interval', '-i', 
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
                       default='15m',
                       help='캔들 시간대 (기본값: 15m)')
    parser.add_argument('--volume', '-v',
                       type=int,
                       default=1000000,
                       help='최소 거래량 기준 USDT (기본값: 1000000)')
    parser.add_argument('--test', '-t',
                       action='store_true',
                       help='테스트 모드 (즉시 한 번 실행)')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 테스트 모드로 즉시 실행합니다...")
        scanner = TradingStrategyScanner(interval=args.interval, volume_threshold=args.volume)
        scanner.run_scanner()
        return
    
    print("🚀 스케줄러를 시작합니다...")
    print("📅 매 시각 10분, 25분, 40분, 55분마다 스캔을 실행합니다.")
    print("⏰ 현재 시간:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("🛑 중지하려면 Ctrl+C를 누르세요.")
    print("-" * 60)
    
    # 스케줄 설정
    schedule.every().hour.at(":10").do(run_scheduled_scan)
    schedule.every().hour.at(":25").do(run_scheduled_scan)
    schedule.every().hour.at(":40").do(run_scheduled_scan)
    schedule.every().hour.at(":55").do(run_scheduled_scan)
    
    # 스케줄러 실행
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 스케줄러를 종료합니다.")

if __name__ == "__main__":
    main()
