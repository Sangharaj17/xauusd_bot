"""
XAUUSD Trading Signal Generator - Fixed Version
Key fixes:
1. Proper parameter handling for Supabase credentials
2. Better error handling and logging
3. Fallback mechanisms for data fetching
4. Debug mode for troubleshooting
"""

import requests
import json
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import argparse
from supabase import create_client, Client

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XAUUSDSignalGenerator:
    """Main class for generating XAUUSD trading signals"""
    
    def __init__(
        self, 
        alpha_vantage_key: str = "demo", 
        mode: str = "standard",
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        debug: bool = False
    ):
        self.alpha_vantage_key = alpha_vantage_key
        self.tp_pips = 40
        self.sl_pips = 60
        self.pip_value = 0.10
        self.signal_history = []
        self.mode = mode if mode in ("standard", "aggressive") else "standard"
        self.debug = debug
        
        # Telegram credentials
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        # Initialize Supabase - FIXED: Don't override parameters
        self.supabase_client = None
        if supabase_url and supabase_key:
            try:
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase client initialized successfully")
                
                # Test connection
                if self.debug:
                    test = self.supabase_client.table('signal_history').select("id").limit(1).execute()
                    logger.info(f"✅ Supabase connection test passed: {len(test.data) if test.data else 0} records found")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase: {e}")
                if self.debug:
                    import traceback
                    logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️  Supabase credentials not provided - database features disabled")
    
    def fetch_market_data(self, symbol: str = "GC=F", interval: str = "5m") -> pd.DataFrame:
        """Fetch OHLC data from Yahoo Finance with better error handling"""
        try:
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m"}
            yf_interval = interval_map.get(interval, "5m")

            range_map = {"1m": "1d", "5m": "5d", "15m": "1mo", "1h": "3mo"}
            yf_range = range_map.get(interval, "5d")

            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': yf_interval, 'range': yf_range}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            if self.debug:
                logger.info(f"📊 Fetching {symbol} data: interval={yf_interval}, range={yf_range}")

            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if not response.ok:
                logger.error(f"❌ Yahoo Finance request failed: {response.status_code}")
                if self.debug:
                    logger.error(f"Response: {response.text[:200]}")
                return pd.DataFrame()

            data = response.json()
            
            if 'chart' not in data or not data['chart'].get('result'):
                logger.error("❌ Invalid response structure from Yahoo Finance")
                if self.debug:
                    logger.error(f"Response keys: {data.keys()}")
                return pd.DataFrame()
            
            result = data['chart']['result'][0]
            timestamps = result.get('timestamp')
            
            if not timestamps:
                logger.error("❌ No timestamp data in response")
                return pd.DataFrame()
            
            quote = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume']
            })
            
            df.dropna(inplace=True)
            
            if self.debug:
                logger.info(f"✅ Fetched {len(df)} candles, latest price: {df['close'].iloc[-1]:.2f}")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> float:
        """Calculate RSI with error handling"""
        try:
            if len(data) < period + 1:
                logger.warning(f"⚠️  Insufficient data for RSI calculation: {len(data)} < {period + 1}")
                return 50.0
                
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero
            if loss.iloc[-1] == 0:
                return 100.0
                
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.error(f"❌ RSI calculation error: {e}")
            return 50.0
    
    def calculate_macd(self, data: pd.Series) -> Dict[str, float]:
        """Calculate MACD with error handling"""
        try:
            if len(data) < 26:
                logger.warning(f"⚠️  Insufficient data for MACD: {len(data)} < 26")
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'crossover': None}
                
            ema_fast = data.ewm(span=12, adjust=False).mean()
            ema_slow = data.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            macd_cross = None
            if len(macd_line) > 1:
                if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                    macd_cross = "bullish"
                elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                    macd_cross = "bearish"
            
            return {
                'macd': float(macd_line.iloc[-1]),
                'signal': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1]),
                'crossover': macd_cross
            }
        except Exception as e:
            logger.error(f"❌ MACD calculation error: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'crossover': None}
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands with error handling"""
        try:
            if len(data) < period:
                logger.warning(f"⚠️  Insufficient data for BB: {len(data)} < {period}")
                return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'position': None}
                
            sma = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            current_price = float(data.iloc[-1])
            bb_position = None
            if current_price < float(lower.iloc[-1]):
                bb_position = "below_lower"
            elif current_price > float(upper.iloc[-1]):
                bb_position = "above_upper"
            
            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower.iloc[-1]),
                'position': bb_position
            }
        except Exception as e:
            logger.error(f"❌ BB calculation error: {e}")
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'position': None}
    
    def calculate_moving_averages(self, data: pd.Series) -> Dict[str, Optional[float]]:
        """Calculate moving averages with error handling"""
        try:
            mas = {}
            if len(data) >= 9:
                mas['ema_9'] = float(data.ewm(span=9, adjust=False).mean().iloc[-1])
            if len(data) >= 21:
                mas['ema_21'] = float(data.ewm(span=21, adjust=False).mean().iloc[-1])
            if len(data) >= 20:
                mas['sma_20'] = float(data.rolling(window=20).mean().iloc[-1])
            if len(data) >= 50:
                mas['sma_50'] = float(data.rolling(window=50).mean().iloc[-1])
            return mas
        except Exception as e:
            logger.error(f"❌ MA calculation error: {e}")
            return {}
    
    def analyze_timeframe(self, interval: str) -> Dict:
        """Analyze single timeframe with enhanced logging"""
        if self.debug:
            logger.info(f"🔍 Analyzing {interval} timeframe...")
            
        df = self.fetch_market_data(interval=interval)
        
        if df.empty or len(df) < 30:
            logger.warning(f"⚠️  {interval}: Insufficient data ({len(df)} candles)")
            return {'valid': False, 'interval': interval, 'reason': 'insufficient_data'}
        
        close = df['close']
        current_price = float(close.iloc[-1])
        
        # Calculate indicators
        rsi = self.calculate_rsi(close)
        macd = self.calculate_macd(close)
        bb = self.calculate_bollinger_bands(close)
        mas = self.calculate_moving_averages(close)
        
        # Determine trend
        trend = "neutral"
        if 'ema_9' in mas and 'ema_21' in mas:
            if mas['ema_9'] > mas['ema_21']:
                trend = "bullish"
            elif mas['ema_9'] < mas['ema_21']:
                trend = "bearish"
        
        # Collect signals
        signals = []
        if rsi < 30:
            signals.append("RSI_OVERSOLD")
        elif rsi > 70:
            signals.append("RSI_OVERBOUGHT")
        elif self.mode == "aggressive":
            if rsi < 35:
                signals.append("RSI_OVERSOLD_WEAK")
            elif rsi > 65:
                signals.append("RSI_OVERBOUGHT_WEAK")
        
        if macd['crossover'] == 'bullish':
            signals.append("MACD_BULLISH_CROSS")
        elif macd['crossover'] == 'bearish':
            signals.append("MACD_BEARISH_CROSS")
        
        if bb['position'] == 'below_lower':
            signals.append("BB_LOWER_BREACH")
        elif bb['position'] == 'above_upper':
            signals.append("BB_UPPER_BREACH")
        
        if self.debug:
            logger.info(f"✅ {interval}: Price={current_price:.2f}, RSI={rsi:.1f}, Trend={trend}, Signals={signals}")
        
        return {
            'valid': True,
            'interval': interval,
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'bollinger_bands': bb,
            'moving_averages': mas,
            'trend': trend,
            'signals': signals
        }
    
    def determine_direction(self, m5: Dict, m15: Dict, h1: Dict, sentiment: Dict) -> Dict[str, Optional[float]]:
        """Determine trade direction with debug info"""
        bullish_score = 0.0
        bearish_score = 0.0
        
        # M5 signals (weight: 2)
        if 'RSI_OVERSOLD' in m5['signals'] or 'BB_LOWER_BREACH' in m5['signals']:
            bullish_score += 2
        if 'MACD_BULLISH_CROSS' in m5['signals']:
            bullish_score += 2
        if 'RSI_OVERBOUGHT' in m5['signals'] or 'BB_UPPER_BREACH' in m5['signals']:
            bearish_score += 2
        if 'MACD_BEARISH_CROSS' in m5['signals']:
            bearish_score += 2

        # Aggressive mode bonuses
        if self.mode == "aggressive":
            if 'RSI_OVERSOLD_WEAK' in m5['signals']:
                bullish_score += 1
            if 'RSI_OVERBOUGHT_WEAK' in m5['signals']:
                bearish_score += 1
            if isinstance(m5.get('macd'), dict):
                hist = m5['macd'].get('histogram', 0)
                if hist > 0:
                    bullish_score += 0.5
                elif hist < 0:
                    bearish_score += 0.5
        
        # Trend alignment
        if m15['trend'] == 'bullish':
            bullish_score += 1
        elif m15['trend'] == 'bearish':
            bearish_score += 1
        
        if h1.get('valid') and h1['trend'] == 'bullish':
            bullish_score += 1
        elif h1.get('valid') and h1['trend'] == 'bearish':
            bearish_score += 1
        
        # Sentiment
        if sentiment['label'] == 'bullish':
            bullish_score += 0.5
        elif sentiment['label'] == 'bearish':
            bearish_score += 0.5
        
        if self.debug:
            logger.info(f"📊 Scoring: Bullish={bullish_score:.1f}, Bearish={bearish_score:.1f} (Mode: {self.mode})")
        
        # Decision
        threshold = 2.0 if self.mode == "aggressive" else 3.0
        margin = 0.25 if self.mode == "aggressive" else 1.0
        
        if bullish_score >= threshold and bullish_score > bearish_score + margin:
            return {"direction": "BUY", "bullish_score": bullish_score, "bearish_score": bearish_score}
        if bearish_score >= threshold and bearish_score > bullish_score + margin:
            return {"direction": "SELL", "bullish_score": bullish_score, "bearish_score": bearish_score}
        
        return {"direction": None, "bullish_score": bullish_score, "bearish_score": bearish_score}
    
    def save_signal_to_supabase(self, signal: Dict, telegram_sent: bool = False) -> Optional[int]:
        """Save signal to Supabase with better error handling"""
        if not self.supabase_client:
            logger.warning("⚠️  Supabase client not initialized")
            return None
        
        if not signal.get('direction'):
            if self.debug:
                logger.info("ℹ️  No direction - skipping database save")
            return None
        
        try:
            record = {
                'pair': signal.get('pair', 'XAUUSD'),
                'signal_time': signal.get('time'),
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'take_profit': signal.get('take_profit'),
                'stop_loss': signal.get('stop_loss'),
                'confidence': signal.get('confidence'),
                'status': signal.get('status', 'active'),
                'mode': signal.get('mode', self.mode),
                'telegram_sent': telegram_sent,
                'created_at': datetime.now(UTC).isoformat()
            }
            
            # Add reasoning fields if they exist in your schema
            reasoning = signal.get('reasoning', {})
            if reasoning:
                record['reasoning_technical'] = reasoning.get('technical')
                record['reasoning_fundamental'] = reasoning.get('fundamental')
                record['reasoning_mtf'] = reasoning.get('multi_timeframe')
            
            if signal.get('notes'):
                record['notes'] = signal['notes']
            
            if self.debug:
                logger.info(f"💾 Inserting record: {json.dumps(record, indent=2)}")
            
            response = self.supabase_client.table('signal_history').insert(record).execute()
            
            if response.data and len(response.data) > 0:
                signal_id = response.data[0].get('id')
                logger.info(f"✅ Signal saved to Supabase ID: {signal_id}")
                return signal_id
            else:
                logger.error("❌ Supabase insert returned no data")
                return None
            
        except Exception as e:
            logger.error(f"❌ Supabase save error: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return None
    
    def send_telegram_message(self, text: str) -> bool:
        """Send message to Telegram with error handling"""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                logger.warning("⚠️  Telegram not configured")
                return False
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {'chat_id': self.telegram_chat_id, 'text': text}
            
            if self.debug:
                logger.info(f"📱 Sending to Telegram chat {self.telegram_chat_id}")
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.ok:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram send failed: {response.status_code}")
                if self.debug:
                    logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
    
    def _format_signal_text(self, signal: Dict) -> str:
        """Format signal for Telegram"""
        direction = signal.get('direction')
        confidence = signal.get('confidence', 0)
        entry = signal.get('entry_price', 0)
        tp = signal.get('take_profit', 0)
        sl = signal.get('stop_loss', 0)
        
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        text = f"{emoji} *XAUUSD Signal*\n\n"
        text += f"📊 Direction: *{direction}*\n"
        text += f"💰 Entry: {entry:.2f}\n"
        text += f"🎯 TP: {tp:.2f}\n"
        text += f"🛑 SL: {sl:.2f}\n"
        text += f"📈 Confidence: {confidence:.0%}\n"
        text += f"⚙️  Mode: {signal.get('mode', 'standard')}"
        
        return text
    
    def generate_signal(self) -> Dict:
        """Generate trading signal with enhanced debugging"""
        try:
            logger.info("=" * 60)
            logger.info("🚀 Starting signal generation...")
            logger.info("=" * 60)
            
            # Analyze timeframes
            m5 = self.analyze_timeframe('5m')
            m15 = self.analyze_timeframe('15m')
            h1 = self.analyze_timeframe('1h')
            
            # Validate
            if not m5.get('valid'):
                logger.error("❌ M5 data invalid")
                return {
                    'status': 'error',
                    'reason': 'M5 data unavailable',
                    'timestamp': datetime.now(UTC).isoformat()
                }
            
            if not m15.get('valid'):
                logger.error("❌ M15 data invalid")
                return {
                    'status': 'error',
                    'reason': 'M15 data unavailable',
                    'timestamp': datetime.now(UTC).isoformat()
                }
            
            # Sentiment (simplified for reliability)
            sentiment = {'score': 0.0, 'label': 'neutral', 'description': 'Neutral sentiment'}
            
            # Determine direction
            decision = self.determine_direction(m5, m15, h1, sentiment)
            direction = decision.get('direction')
            
            if direction is None:
                logger.info("ℹ️  No signal generated (insufficient score)")
                return {
                    'status': 'no_signal',
                    'reason': 'No clear opportunity',
                    'timestamp': datetime.now(UTC).isoformat(),
                    'scores': {
                        'bullish': decision.get('bullish_score'),
                        'bearish': decision.get('bearish_score')
                    }
                }
            
            # Calculate prices
            entry_price = m5['price']
            tp_pips = self.tp_pips if self.mode == "standard" else int(self.tp_pips * 0.75)
            sl_pips = self.sl_pips if self.mode == "standard" else int(self.sl_pips * 0.8)
            
            if direction == "BUY":
                take_profit = entry_price + (tp_pips * self.pip_value)
                stop_loss = entry_price - (sl_pips * self.pip_value)
            else:
                take_profit = entry_price - (tp_pips * self.pip_value)
                stop_loss = entry_price + (sl_pips * self.pip_value)
            
            # Build signal
            signal = {
                'pair': 'XAUUSD',
                'time': datetime.now(UTC).isoformat(),
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'take_profit': round(take_profit, 2),
                'stop_loss': round(stop_loss, 2),
                'confidence': 0.75,  # Simplified
                'status': 'active',
                'mode': self.mode,
                'reasoning': {
                    'technical': f"M5 signals: {', '.join(m5['signals'])}",
                    'multi_timeframe': f"M15: {m15['trend']}, H1: {h1.get('trend', 'N/A')}"
                }
            }
            
            logger.info(f"✅ Signal generated: {direction} at {entry_price:.2f}")
            
            # Send to Telegram
            telegram_sent = False
            if self.telegram_token and self.telegram_chat_id:
                msg = self._format_signal_text(signal)
                telegram_sent = self.send_telegram_message(msg)
            
            # Save to database
            db_id = self.save_signal_to_supabase(signal, telegram_sent)
            
            signal['telegram_sent'] = telegram_sent
            signal['database_saved'] = db_id is not None
            if db_id:
                signal['database_id'] = db_id
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Fatal error in generate_signal: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'reason': str(e),
                'timestamp': datetime.now(UTC).isoformat()
            }


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Signal Generator - Fixed Version")
    parser.add_argument('--mode', default='aggressive', choices=['standard', 'aggressive'])
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--telegram-token', help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', help='Telegram chat ID')
    parser.add_argument('--supabase-url', help='Supabase project URL')
    parser.add_argument('--supabase-key', help='Supabase anon/service key')
    args = parser.parse_args()
    
    # Use environment variables as fallback
    import os
    telegram_token = args.telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = args.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
    supabase_url = args.supabase_url or os.getenv('SUPABASE_URL')
    supabase_key = args.supabase_key or os.getenv('SUPABASE_KEY')
    
    # Initialize generator
    generator = XAUUSDSignalGenerator(
        alpha_vantage_key="demo",
        mode=args.mode,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        debug=args.debug
    )
    
    # Generate signal
    signal = generator.generate_signal()
    print("\n" + "=" * 60)
    print("SIGNAL OUTPUT")
    print("=" * 60)
    print(json.dumps(signal, indent=2))


if __name__ == "__main__":
    main()