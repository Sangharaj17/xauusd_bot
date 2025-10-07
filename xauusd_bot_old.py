"""
XAUUSD Trading Signal Generator - JSON Output Only
Returns signals in pure JSON format for API integration
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseSignalStorage:
    """Handles storing and retrieving signals from Supabase"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase anon/service key
        """
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def save_signal(self, signal: Dict, telegram_sent: bool = False) -> Optional[Dict]:
        """
        Save a trading signal to Supabase
        
        Args:
            signal: The signal dictionary from generator
            telegram_sent: Whether the signal was successfully sent to Telegram
            
        Returns:
            The inserted record or None if failed
        """
        try:
            # Prepare data for insertion
            record = {
                'pair': signal.get('pair', 'XAUUSD'),
                'signal_time': signal.get('time'),
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'take_profit': signal.get('take_profit'),
                'stop_loss': signal.get('stop_loss'),
                'confidence': signal.get('confidence'),
                'status': signal.get('status', 'active'),
                'mode': signal.get('mode', 'standard'),
                'telegram_sent': telegram_sent,
                'reasoning_technical': signal.get('reasoning', {}).get('technical'),
                'reasoning_fundamental': signal.get('reasoning', {}).get('fundamental'),
                'reasoning_mtf': signal.get('reasoning', {}).get('multi_timeframe'),
                'notes': signal.get('notes'),
                'created_at': datetime.now(UTC).isoformat()
            }
            
            # Insert into Supabase
            response = self.client.table('signal_history').insert(record).execute()
            
            if response.data:
                logger.info(f"Signal saved to Supabase: ID {response.data[0].get('id')}")
                return response.data[0]
            else:
                logger.error("No data returned from Supabase insert")
                return None
                
        except Exception as e:
            logger.error(f"Error saving signal to Supabase: {e}")
            return None
    
    def update_signal_outcome(self, signal_id: int, outcome: str, 
                             close_price: Optional[float] = None,
                             close_time: Optional[str] = None,
                             profit_loss: Optional[float] = None) -> bool:
        """
        Update signal outcome after trade closes
        
        Args:
            signal_id: The database ID of the signal
            outcome: 'win', 'loss', or 'breakeven'
            close_price: The closing price
            close_time: When the trade closed
            profit_loss: Profit/loss in pips or currency
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {
                'outcome': outcome,
                'status': 'closed',
                'updated_at': datetime.now(UTC).isoformat()
            }
            
            if close_price is not None:
                update_data['close_price'] = close_price
            if close_time is not None:
                update_data['close_time'] = close_time
            if profit_loss is not None:
                update_data['profit_loss'] = profit_loss
            
            response = self.client.table('signal_history')\
                .update(update_data)\
                .eq('id', signal_id)\
                .execute()
            
            if response.data:
                logger.info(f"Signal {signal_id} updated with outcome: {outcome}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")
            return False
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent signals from database
        
        Args:
            limit: Number of signals to retrieve
            
        Returns:
            List of signal records
        """
        try:
            response = self.client.table('signal_history')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error fetching recent signals: {e}")
            return []
    
    def get_performance_stats(self, days: int = 7) -> Dict:
        """
        Calculate performance statistics from stored signals
        
        Args:
            days: Number of days to include in statistics
            
        Returns:
            Performance metrics dictionary
        """
        try:
            cutoff_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            
            response = self.client.table('signal_history')\
                .select('*')\
                .gte('created_at', cutoff_date)\
                .execute()
            
            signals = response.data if response.data else []
            
            total = len(signals)
            closed = [s for s in signals if s.get('outcome') in ['win', 'loss', 'breakeven']]
            wins = len([s for s in closed if s.get('outcome') == 'win'])
            losses = len([s for s in closed if s.get('outcome') == 'loss'])
            
            win_rate = (wins / len(closed)) if closed else 0.0
            avg_confidence = sum(s.get('confidence', 0) for s in signals) / total if total else 0.0
            
            return {
                'total_signals': total,
                'closed_signals': len(closed),
                'active_signals': total - len(closed),
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 4),
                'average_confidence': round(avg_confidence, 4),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {}

class XAUUSDSignalGenerator:
    """Main class for generating XAUUSD trading signals in JSON format"""
    
    def __init__(self, alpha_vantage_key: str = "demo", mode: str = "standard", telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.tp_pips = 40
        self.sl_pips = 60
        self.pip_value = 0.10
        self.signal_history = []
        # modes: "standard" (conservative), "aggressive" (more signals)
        self.mode = mode if mode in ("standard", "aggressive") else "standard"
        # Telegram (direct-only, no env fallback)
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        

        self.supabase_client = None
        supabase_url = "https://bkbfirqkrwomlxuuvhnz.supabase.co" 
        supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJrYmZpcnFrcndvbWx4dXV2aG56Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc5NDkwMDMsImV4cCI6MjA3MzUyNTAwM30.PjMUeIMhoS_ZVM0xfueVystaJkJ3hrNj954cuJb1ylE"  # Your anon or service key

        if supabase_url and supabase_key:
            try:
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
    
    def fetch_market_data(self, symbol: str = "GC=F", interval: str = "5m") -> pd.DataFrame:
        """Fetch OHLC data from Yahoo Finance"""
        try:
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m"}
            yf_interval = interval_map.get(interval, "5m")

            # Choose appropriate range to ensure >= 30 candles
            range_map = {
                "1m": "1d",
                "5m": "5d",
                "15m": "1mo",
                "1h": "3mo",
            }
            yf_range = range_map.get(interval, "5d")

            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': yf_interval, 'range': yf_range}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*'
            }

            response = requests.get(url, params=params, headers=headers, timeout=15)
            if not response.ok:
                logger.error(f"Yahoo data request failed: {response.status_code} {response.text[:120]}")
                return pd.DataFrame()

            try:
                data = response.json()
            except Exception as parse_err:
                logger.error(f"Failed to parse Yahoo response as JSON: {parse_err}; body head: {response.text[:120]}")
                return pd.DataFrame()
            
            if 'chart' in data and 'result' in data['chart']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
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
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_market_data_range(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLC data for an explicit time range by stitching Yahoo 'period1/period2' API."""
        try:
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m"}
            yf_interval = interval_map.get(interval, "5m")
            # Yahoo expects unix seconds
            p1 = int(start.timestamp())
            p2 = int(end.timestamp())
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': p1,
                'period2': p2,
                'interval': yf_interval
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*'
            }
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if not response.ok:
                logger.error(f"Yahoo range request failed: {response.status_code} {response.text[:120]}")
                return pd.DataFrame()
            data = response.json()
            if 'chart' not in data or not data['chart'].get('result'):
                return pd.DataFrame()
            result = data['chart']['result'][0]
            timestamps = result.get('timestamp')
            if not timestamps:
                return pd.DataFrame()
            quote = result['indicators']['quote'][0]
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s', utc=True),
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume']
            })
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching range data: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    def calculate_macd(self, data: pd.Series) -> Dict[str, float]:
        """Calculate MACD indicator"""
        try:
            ema_fast = data.ewm(span=12).mean()
            ema_slow = data.ewm(span=26).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # Detect crossover
            macd_cross = None
            if len(macd_line) > 1 and len(signal_line) > 1:
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
        except:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'crossover': None}
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
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
        except:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'position': None}
    
    def calculate_moving_averages(self, data: pd.Series) -> Dict[str, Optional[float]]:
        """Calculate moving averages"""
        try:
            mas = {}
            if len(data) >= 9:
                mas['ema_9'] = float(data.ewm(span=9).mean().iloc[-1])
            if len(data) >= 21:
                mas['ema_21'] = float(data.ewm(span=21).mean().iloc[-1])
            if len(data) >= 20:
                mas['sma_20'] = float(data.rolling(window=20).mean().iloc[-1])
            if len(data) >= 50:
                mas['sma_50'] = float(data.rolling(window=50).mean().iloc[-1])
            return mas
        except:
            return {}
    
    def fetch_news_sentiment(self) -> Dict[str, Union[float, str]]:
        """Fetch market sentiment from news"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': 'FOREX:USD',
                'topics': 'economy_fiscal,economy_monetary',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            sentiment_score = 0.0
            sentiment_label = "neutral"
            
            if 'feed' in data and len(data['feed']) > 0:
                sentiments = [float(item.get('overall_sentiment_score', 0)) 
                             for item in data['feed'][:5]]
                sentiment_score = float(np.mean(sentiments))
                
                if sentiment_score > 0.15:
                    sentiment_label = "bullish"
                elif sentiment_score < -0.15:
                    sentiment_label = "bearish"
            
            return {
                'score': sentiment_score,
                'label': sentiment_label,
                'description': f"Market sentiment: {sentiment_label} ({sentiment_score:.2f})"
            }
        
        except Exception as e:
            logger.warning(f"News sentiment unavailable: {e}")
            return {
                'score': 0.0,
                'label': 'neutral',
                'description': 'Market sentiment: neutral (no data)'
            }
    
    def analyze_timeframe(self, interval: str) -> Dict:
        """Analyze single timeframe"""
        df = self.fetch_market_data(interval=interval)
        
        if df.empty or len(df) < 30:
            return {'valid': False, 'interval': interval}
        
        close = df['close']
        current_price = float(close.iloc[-1])
        
        # Calculate all indicators
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
        else:
            # Weaker RSI signals in aggressive mode
            if self.mode == "aggressive":
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

    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """Analyze indicators from a provided OHLCV DataFrame (expects columns: open, high, low, close, volume)."""
        if df is None or df.empty or len(df) < 30:
            return {'valid': False}
        close = df['close']
        current_price = float(close.iloc[-1])
        rsi = self.calculate_rsi(close)
        macd = self.calculate_macd(close)
        bb = self.calculate_bollinger_bands(close)
        mas = self.calculate_moving_averages(close)
        trend = "neutral"
        if 'ema_9' in mas and 'ema_21' in mas:
            if mas['ema_9'] > mas['ema_21']:
                trend = "bullish"
            elif mas['ema_9'] < mas['ema_21']:
                trend = "bearish"
        signals = []
        if rsi < 30:
            signals.append("RSI_OVERSOLD")
        elif rsi > 70:
            signals.append("RSI_OVERBOUGHT")
        else:
            if self.mode == "aggressive":
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
        return {
            'valid': True,
            'price': current_price,
            'rsi': rsi,
            'macd': macd,
            'bollinger_bands': bb,
            'moving_averages': mas,
            'trend': trend,
            'signals': signals
        }
    
    def is_high_impact_news_time(self) -> bool:
        """Check if during high-impact news"""
        now = datetime.now(UTC)
        # Friday NFP time example (12:30-14:00 UTC)
        if now.weekday() == 4 and 12 <= now.hour < 14:
            return True
        return False
    
    def determine_direction(self, m5: Dict, m15: Dict, h1: Dict, sentiment: Dict) -> Dict[str, Optional[float]]:
        """Determine trade direction and expose scoring details"""
        bullish_score: float = 0
        bearish_score: float = 0
        
        # M5 signals (weight: 2)
        if 'RSI_OVERSOLD' in m5['signals'] or 'BB_LOWER_BREACH' in m5['signals']:
            bullish_score += 2
        if 'MACD_BULLISH_CROSS' in m5['signals']:
            bullish_score += 2
        if 'RSI_OVERBOUGHT' in m5['signals'] or 'BB_UPPER_BREACH' in m5['signals']:
            bearish_score += 2
        if 'MACD_BEARISH_CROSS' in m5['signals']:
            bearish_score += 2

        # Weaker confirmations in aggressive mode (weight: 1)
        if self.mode == "aggressive":
            if 'RSI_OVERSOLD_WEAK' in m5['signals']:
                bullish_score += 1
            if 'RSI_OVERBOUGHT_WEAK' in m5['signals']:
                bearish_score += 1
            # MACD histogram bias
            if isinstance(m5.get('macd'), dict):
                hist = m5['macd'].get('histogram')
                if isinstance(hist, (int, float)):
                    if hist > 0:
                        bullish_score += 0.5
                    elif hist < 0:
                        bearish_score += 0.5
        
        # M15 trend (weight: 1)
        if m15['trend'] == 'bullish':
            bullish_score += 1
        elif m15['trend'] == 'bearish':
            bearish_score += 1
        
        # H1 trend (weight: 1)
        if h1.get('valid') and h1['trend'] == 'bullish':
            bullish_score += 1
        elif h1.get('valid') and h1['trend'] == 'bearish':
            bearish_score += 1
        
        # Sentiment (weight: 0.5)
        if sentiment['label'] == 'bullish':
            bullish_score += 0.5
        elif sentiment['label'] == 'bearish':
            bearish_score += 0.5
        
        # Decision logic
        if self.mode == "aggressive":
            if bullish_score >= 2.0 and bullish_score > bearish_score + 0.25:
                return {"direction": "BUY", "bullish_score": bullish_score, "bearish_score": bearish_score}
            if bearish_score >= 2.0 and bearish_score > bullish_score + 0.25:
                return {"direction": "SELL", "bullish_score": bullish_score, "bearish_score": bearish_score}
        else:
            if bullish_score >= 3 and bullish_score > bearish_score + 1:
                return {"direction": "BUY", "bullish_score": bullish_score, "bearish_score": bearish_score}
            if bearish_score >= 3 and bearish_score > bullish_score + 1:
                return {"direction": "SELL", "bullish_score": bullish_score, "bearish_score": bearish_score}

        return {"direction": None, "bullish_score": bullish_score, "bearish_score": bearish_score}
    
    def calculate_confidence(self, m5: Dict, m15: Dict, h1: Dict, sentiment: Dict, direction: str) -> float:
        """Calculate signal confidence"""
        confidence = 0.45 if self.mode == "aggressive" else 0.5
        
        # RSI confirmation
        if direction == "BUY" and (m5['rsi'] < (37 if self.mode == "aggressive" else 35)):
            confidence += 0.15
        elif direction == "SELL" and (m5['rsi'] > (63 if self.mode == "aggressive" else 65)):
            confidence += 0.15
        
        # MACD confirmation
        if direction == "BUY" and m5['macd']['crossover'] == 'bullish':
            confidence += 0.1
        elif direction == "SELL" and m5['macd']['crossover'] == 'bearish':
            confidence += 0.1
        
        # Trend alignment
        if direction == "BUY" and m15['trend'] == 'bullish':
            confidence += 0.1
        elif direction == "SELL" and m15['trend'] == 'bearish':
            confidence += 0.1
        
        # Sentiment confirmation
        if direction == "BUY" and sentiment['label'] == 'bullish':
            confidence += 0.1
        elif direction == "SELL" and sentiment['label'] == 'bearish':
            confidence += 0.1
        elif sentiment['label'] != 'neutral':
            if (direction == "BUY" and sentiment['label'] == 'bearish') or \
               (direction == "SELL" and sentiment['label'] == 'bullish'):
                confidence -= 0.15
        
        # Slightly penalize when M15 and H1 disagree in aggressive mode
        if self.mode == "aggressive" and h1.get('valid') and h1['trend'] != 'neutral' and m15['trend'] != 'neutral':
            if h1['trend'] != m15['trend']:
                confidence -= 0.05

        return round(min(max(confidence, 0.0), 1.0), 2)
    
    def generate_reasoning(self, m5: Dict, m15: Dict, h1: Dict, sentiment: Dict) -> Dict[str, str]:
        """Generate reasoning for the signal"""
        technical_reasons = []
        
        if m5['rsi'] < 35:
            technical_reasons.append(f"RSI oversold ({m5['rsi']:.1f})")
        elif m5['rsi'] > 65:
            technical_reasons.append(f"RSI overbought ({m5['rsi']:.1f})")
        
        if m5['macd']['crossover'] == 'bullish':
            technical_reasons.append("MACD bullish crossover")
        elif m5['macd']['crossover'] == 'bearish':
            technical_reasons.append("MACD bearish crossover")
        
        if 'BB_LOWER_BREACH' in m5['signals']:
            technical_reasons.append("Price below lower Bollinger Band")
        elif 'BB_UPPER_BREACH' in m5['signals']:
            technical_reasons.append("Price above upper Bollinger Band")
        
        mtf_reasons = []
        if m15['trend'] != 'neutral':
            mtf_reasons.append(f"M15 {m15['trend']} trend")
        if h1.get('valid') and h1['trend'] != 'neutral':
            mtf_reasons.append(f"H1 {h1['trend']} trend")
        
        return {
            'technical': ', '.join(technical_reasons) if technical_reasons else "Multiple technical indicators aligned",
            'fundamental': sentiment['description'],
            'multi_timeframe': ', '.join(mtf_reasons) if mtf_reasons else "Multi-timeframe analysis completed"
        }
    


    def save_signal_to_supabase(self, signal: Dict, telegram_sent: bool = False) -> Optional[int]:
        """
        Save signal to Supabase database
        
        Args:
            signal: The signal dictionary
            telegram_sent: Whether the signal was successfully sent to Telegram
        
        Returns:
            Signal ID if successful, None otherwise
        """
        if not self.supabase_client:
            logger.warning("Supabase client not initialized - signal not saved to database")
            return None
        
        try:
            # Only save signals that have a valid direction (actual trading signals)
            if not signal.get('direction'):
                logger.info("No trading direction - skipping database save")
                return None
            
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
                'reasoning_technical': signal.get('reasoning', {}).get('technical'),
                'reasoning_fundamental': signal.get('reasoning', {}).get('fundamental'),
                'reasoning_mtf': signal.get('reasoning', {}).get('multi_timeframe'),
                'notes': signal.get('notes'),
                'created_at': datetime.now(UTC).isoformat()
            }
            
            response = self.supabase_client.table('signal_history').insert(record).execute()
            
            if response.data and len(response.data) > 0:
                signal_id = response.data[0].get('id')
                logger.info(f"✅ Signal saved to Supabase with ID: {signal_id} (Telegram sent: {telegram_sent})")
                return signal_id
            else:
                logger.warning("Supabase insert returned no data")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error saving to Supabase: {e}")
            return None
    
    def get_recent_signals_from_db(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve recent signals from Supabase
        
        Args:
            limit: Number of signals to retrieve
            
        Returns:
            List of signal records
        """
        if not self.supabase_client:
            logger.warning("Supabase client not initialized")
            return []
        
        try:
            response = self.supabase_client.table('signal_history')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error fetching recent signals: {e}")
            return []
    
    def get_performance_from_db(self, days: int = 7) -> Dict:
        """
        Calculate performance statistics from Supabase
        
        Args:
            days: Number of days to include
            
        Returns:
            Performance metrics
        """
        if not self.supabase_client:
            logger.warning("Supabase client not initialized")
            return {}
        
        try:
            cutoff_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            
            response = self.supabase_client.table('signal_history')\
                .select('*')\
                .gte('created_at', cutoff_date)\
                .execute()
            
            signals = response.data if response.data else []
            
            total = len(signals)
            closed = [s for s in signals if s.get('outcome') in ['win', 'loss', 'breakeven']]
            wins = len([s for s in closed if s.get('outcome') == 'win'])
            losses = len([s for s in closed if s.get('outcome') == 'loss'])
            telegram_sent_count = len([s for s in signals if s.get('telegram_sent')])
            
            win_rate = (wins / len(closed)) if closed else 0.0
            avg_confidence = sum(s.get('confidence', 0) for s in signals) / total if total else 0.0
            
            return {
                'period_days': days,
                'total_signals': total,
                'telegram_sent': telegram_sent_count,
                'closed_signals': len(closed),
                'active_signals': total - len(closed),
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 4),
                'average_confidence': round(avg_confidence, 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    # def generate_signal(self) -> Dict:
    #     """
    #     Generate a single trading signal
    #     Returns: JSON-formatted signal or error
    #     """
    #     try:
    #         # Check for high-impact news
    #         if self.is_high_impact_news_time():
    #             return {
    #                 'status': 'suppressed',
    #                 'reason': 'High-impact news time detected',
    #                 'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z')
    #             }
            
    #         # Fetch multi-timeframe data
    #         m5 = self.analyze_timeframe('5m')
    #         m15 = self.analyze_timeframe('15m')
    #         h1 = self.analyze_timeframe('1h')
            
    #         # Validate data
    #         if not m5.get('valid') or not m15.get('valid'):
    #             return {
    #                 'status': 'error',
    #                 'reason': 'Insufficient market data',
    #                 'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z')
    #             }
            
    #         # Fetch sentiment
    #         sentiment = self.fetch_news_sentiment()
            
    #         # Determine direction
    #         decision = self.determine_direction(m5, m15, h1, sentiment)
    #         direction = decision.get('direction')

    #         if direction is None:
    #             return {
    #                 'status': 'no_signal',
    #                 'reason': 'No clear trading opportunity detected',
    #                 'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z'),
    #                 'analysis': {
    #                     'm5_signals': m5['signals'],
    #                     'm15_trend': m15['trend'],
    #                     'sentiment': sentiment['label'],
    #                     'scores': {
    #                         'bullish': decision.get('bullish_score'),
    #                         'bearish': decision.get('bearish_score')
    #                     }
    #                 }
    #             }
            
    #         # Calculate entry, TP, SL
    #         entry_price = m5['price']
            
    #         tp_pips = self.tp_pips
    #         sl_pips = self.sl_pips
    #         # In aggressive mode, use tighter SL/TP by default
    #         if self.mode == "aggressive":
    #             tp_pips = int(round(self.tp_pips * 0.75))
    #             sl_pips = int(round(self.sl_pips * 0.8))

    #         if direction == "BUY":
    #             take_profit = entry_price + (tp_pips * self.pip_value)
    #             stop_loss = entry_price - (sl_pips * self.pip_value)
    #         else:
    #             take_profit = entry_price - (tp_pips * self.pip_value)
    #             stop_loss = entry_price + (sl_pips * self.pip_value)
            
    #         # Calculate confidence
    #         confidence = self.calculate_confidence(m5, m15, h1, sentiment, direction)
            
    #         # Generate reasoning
    #         reasoning = self.generate_reasoning(m5, m15, h1, sentiment)
            
    #         # Build signal JSON
    #         signal = {
    #             'pair': 'XAUUSD',
    #             'time': datetime.now(UTC).isoformat().replace('+00:00','Z'),
    #             'direction': direction,
    #             'entry_price': round(entry_price, 2),
    #             'take_profit': round(take_profit, 2),
    #             'stop_loss': round(stop_loss, 2),
    #             'confidence': confidence,
    #             'reasoning': reasoning,
    #             'status': 'active',
    #             'mode': self.mode
    #         }

    #         # In aggressive mode, if confidence is low but near threshold, mark as watchlist
    #         if self.mode == 'aggressive' and confidence < 0.55:
    #             signal['status'] = 'watchlist'
    #             signal['notes'] = 'Near-threshold conditions; monitor for confirmation'
            
    #         # Add to history
    #         self.signal_history.append({
    #             'signal': signal,
    #             'timestamp': datetime.now(UTC),
    #             'outcome': None
    #         })

    #         # Optional Telegram notify
    #         try:
    #             if self.telegram_token and self.telegram_chat_id:
    #                 msg = self._format_signal_text(signal)
    #                 self.send_telegram_message(msg)
    #         except Exception as _:
    #             pass
            
    #         return signal
        
    #     except Exception as e:
    #         logger.error(f"Error generating signal: {e}")
    #         return {
    #             'status': 'error',
    #             'reason': str(e),
    #             'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z')
    #         }


    def generate_signal(self) -> Dict:
        """
        Generate a single trading signal and save to Supabase if sent to Telegram
        Returns: JSON-formatted signal or error
        """
        try:
            # Check for high-impact news
            if self.is_high_impact_news_time():
                return {
                    'status': 'suppressed',
                    'reason': 'High-impact news time detected',
                    'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z')
                }
            
            # Fetch multi-timeframe data
            m5 = self.analyze_timeframe('5m')
            m15 = self.analyze_timeframe('15m')
            h1 = self.analyze_timeframe('1h')
            
            # Validate data
            if not m5.get('valid') or not m15.get('valid'):
                return {
                    'status': 'error',
                    'reason': 'Insufficient market data',
                    'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z')
                }
            
            # Fetch sentiment
            sentiment = self.fetch_news_sentiment()
            
            # Determine direction
            decision = self.determine_direction(m5, m15, h1, sentiment)
            direction = decision.get('direction')

            if direction is None:
                return {
                    'status': 'no_signal',
                    'reason': 'No clear trading opportunity detected',
                    'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z'),
                    'analysis': {
                        'm5_signals': m5['signals'],
                        'm15_trend': m15['trend'],
                        'sentiment': sentiment['label'],
                        'scores': {
                            'bullish': decision.get('bullish_score'),
                            'bearish': decision.get('bearish_score')
                        }
                    }
                }
            
            # Calculate entry, TP, SL
            entry_price = m5['price']
            
            tp_pips = self.tp_pips
            sl_pips = self.sl_pips
            if self.mode == "aggressive":
                tp_pips = int(round(self.tp_pips * 0.75))
                sl_pips = int(round(self.sl_pips * 0.8))

            if direction == "BUY":
                take_profit = entry_price + (tp_pips * self.pip_value)
                stop_loss = entry_price - (sl_pips * self.pip_value)
            else:
                take_profit = entry_price - (tp_pips * self.pip_value)
                stop_loss = entry_price + (sl_pips * self.pip_value)
            
            # Calculate confidence
            confidence = self.calculate_confidence(m5, m15, h1, sentiment, direction)
            
            # Generate reasoning
            reasoning = self.generate_reasoning(m5, m15, h1, sentiment)
            
            # Build signal JSON
            signal = {
                'pair': 'XAUUSD',
                'time': datetime.now(UTC).isoformat().replace('+00:00','Z'),
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'take_profit': round(take_profit, 2),
                'stop_loss': round(stop_loss, 2),
                'confidence': confidence,
                'reasoning': reasoning,
                'status': 'active',
                'mode': self.mode
            }

            if self.mode == 'aggressive' and confidence < 0.55:
                signal['status'] = 'watchlist'
                signal['notes'] = 'Near-threshold conditions; monitor for confirmation'
            
            # Add to history
            self.signal_history.append({
                'signal': signal,
                'timestamp': datetime.now(UTC),
                'outcome': None
            })

            # Send to Telegram
            telegram_sent = False
            try:
                if self.telegram_token and self.telegram_chat_id:
                    msg = self._format_signal_text(signal)
                    telegram_sent = self.send_telegram_message(msg)
                    
                    if telegram_sent:
                        logger.info("📱 Signal sent to Telegram successfully")
                    else:
                        logger.warning("⚠️  Failed to send signal to Telegram")
                        
            except Exception as e:
                logger.error(f"❌ Telegram send error: {e}")
            
            # Save to Supabase (only if we have a valid signal with direction)
            database_id = None
            try:
                database_id = self.save_signal_to_supabase(signal, telegram_sent=telegram_sent)
                if database_id:
                    signal['database_id'] = database_id
                    logger.info(f"💾 Signal saved to database with ID: {database_id}")
                else:
                    logger.warning("⚠️  Signal not saved to database")
                    
            except Exception as e:
                logger.error(f"❌ Database save error: {e}")
            
            # Add status info to signal
            signal['telegram_sent'] = telegram_sent
            signal['database_saved'] = database_id is not None
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                'status': 'error',
                'reason': str(e),
                'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z')
            }
    
    def generate_multiple_signals(self, count: int = 5, interval_minutes: int = 15) -> List[Dict]:
        """
        Generate multiple signals with time intervals
        Returns: List of JSON signals
        """
        signals = []
        
        for i in range(count):
            signal = self.generate_signal()
            signals.append(signal)
            
            if i < count - 1:  # Don't wait after last signal
                logger.info(f"Generated signal {i+1}/{count}, waiting {interval_minutes} minutes...")
                # In real implementation, use time.sleep(interval_minutes * 60)
        
        return signals
    
    def calculate_performance_report(self, period: str = "daily") -> Dict:
        """
        Calculate performance metrics
        Returns: JSON performance report
        """
        if period == "daily":
            cutoff = datetime.now(UTC) - timedelta(days=1)
        elif period == "weekly":
            cutoff = datetime.now(UTC) - timedelta(days=7)
        else:
            cutoff = datetime.now(UTC) - timedelta(days=30)
        
        period_signals = [s for s in self.signal_history if s['timestamp'] >= cutoff]
        
        total_signals = len(period_signals)
        resolved_signals = [s for s in period_signals if s['outcome'] in ['win', 'loss']]
        wins = len([s for s in resolved_signals if s['outcome'] == 'win'])
        losses = len([s for s in resolved_signals if s['outcome'] == 'loss'])
        
        win_rate = (wins / len(resolved_signals)) if resolved_signals else 0.0
        avg_confidence = np.mean([s['signal']['confidence'] for s in period_signals]) if period_signals else 0.0
        
        target_met = total_signals >= (25 if period == "weekly" else 4)
        
        return {
            'report_type': f'{period}_performance',
            'period': period,
            'timestamp': datetime.now(UTC).isoformat().replace('+00:00','Z'),
            'metrics': {
                'total_signals': total_signals,
                'active_signals': total_signals - len(resolved_signals),
                'resolved_signals': len(resolved_signals),
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 4),
                'accuracy': round(win_rate, 4),
                'average_confidence': round(avg_confidence, 4)
            },
            'targets': {
                'signals_target': 25 if period == "weekly" else 4,
                'accuracy_target': 0.90,
                'target_met': target_met
            },
            'status': 'on_track' if target_met and win_rate >= 0.85 else 'needs_improvement'
        }
    
    def get_signal_json(self) -> str:
        """
        Generate and return signal as JSON string
        """
        signal = self.generate_signal()
        return json.dumps(signal, indent=2)
    
    def get_batch_signals_json(self, count: int = 5) -> str:
        """
        Generate multiple signals and return as JSON array string
        """
        signals = self.generate_multiple_signals(count)
        return json.dumps(signals, indent=2)
    
    def get_report_json(self, period: str = "daily") -> str:
        """
        Generate performance report as JSON string
        """
        report = self.calculate_performance_report(period)
        return json.dumps(report, indent=2)

    # ---------------------- Telegram ----------------------
    def _format_signal_text(self, signal: Dict) -> str:
        direction = signal.get('direction')
        status = signal.get('status')
        conf = signal.get('confidence')
        entry = signal.get('entry_price')
        tp = signal.get('take_profit')
        sl = signal.get('stop_loss')
        mode = signal.get('mode', self.mode)
        reasons = signal.get('reasoning', {})
        tech = reasons.get('technical', '')
        mtf = reasons.get('multi_timeframe', '')
        return (
            f"XAUUSD Signal\n"
            f"Direction: {direction}\n"
            f"Status: {status}\n"
            f"Confidence: {conf}\n"
            f"Entry: {entry}\nTP: {tp} | SL: {sl}\n"
            f"Mode: {mode}\n"
            f"Tech: {tech}\n"
            f"MTF: {mtf}"
        )

    def send_telegram_message(self, text: str) -> bool:
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                logger.warning("Telegram not configured: missing token or chat_id")
                return False
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = { 'chat_id': self.telegram_chat_id, 'text': text }
            r = requests.post(url, json=payload, timeout=10)
            if not r.ok:
                logger.error(f"Telegram send failed: {r.status_code} {r.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False


# ============================================================================
# BACKTESTER AND OPTIMIZER
# ============================================================================

class Backtester:
    """Simple bar-by-bar backtester using the existing rules on a single timeframe."""

    def __init__(self, generator: XAUUSDSignalGenerator, interval: str = "5m"):
        self.generator = generator
        self.interval = interval

    def _simulate_trade(self, entry_price: float, direction: str, tp_pips: int, sl_pips: int) -> Dict:
        take_profit = entry_price + (tp_pips * self.generator.pip_value) if direction == 'BUY' else entry_price - (tp_pips * self.generator.pip_value)
        stop_loss = entry_price - (sl_pips * self.generator.pip_value) if direction == 'BUY' else entry_price + (sl_pips * self.generator.pip_value)
        return { 'tp': take_profit, 'sl': stop_loss }

    def run(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty:
            return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'signals': []}

        signals = []
        wins = 0
        losses = 0

        # iterate over bars, using rolling window up to i
        for i in range(30, len(df)):
            window = df.iloc[:i].copy()
            analysis = self.generator.analyze_dataframe(window)
            if not analysis.get('valid'):
                continue

            # Build synthetic m15/h1 as same trend for simplicity in MVP
            m5 = {
                'signals': analysis['signals'],
                'rsi': analysis['rsi'],
                'macd': analysis['macd'],
                'price': analysis['price'],
                'trend': analysis['trend']
            }
            m15 = {'trend': analysis['trend'], 'valid': True}
            h1 = {'trend': analysis['trend'], 'valid': True}
            sentiment = {'label': 'neutral'}

            decision = self.generator.determine_direction(m5, m15, h1, sentiment)
            direction = decision.get('direction')
            if direction is None:
                continue

            entry = float(df['close'].iloc[i-1])
            tp_pips = int(round(self.generator.tp_pips * (0.75 if self.generator.mode == 'aggressive' else 1.0)))
            sl_pips = int(round(self.generator.sl_pips * (0.8 if self.generator.mode == 'aggressive' else 1.0)))
            trade = self._simulate_trade(entry, direction, tp_pips, sl_pips)

            # Check next bar only (very simple fill model)
            bar = df.iloc[i]
            high = float(bar['high'])
            low = float(bar['low'])
            hit_tp = (high >= trade['tp']) if direction == 'BUY' else (low <= trade['tp'])
            hit_sl = (low <= trade['sl']) if direction == 'BUY' else (high >= trade['sl'])

            outcome = None
            if hit_tp and hit_sl:
                # If both hit in same bar, assume worst-case first
                outcome = 'loss'
                losses += 1
            elif hit_tp:
                outcome = 'win'
                wins += 1
            elif hit_sl:
                outcome = 'loss'
                losses += 1
            else:
                # no resolution this bar; skip in MVP
                continue

            signals.append({
                'time': str(bar['timestamp']) if 'timestamp' in bar else None,
                'direction': direction,
                'entry': round(entry, 2),
                'tp': round(trade['tp'], 2),
                'sl': round(trade['sl'], 2),
                'outcome': outcome
            })

        trades = wins + losses
        win_rate = (wins / trades) if trades else 0.0
        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 4),
            'signals': signals
        }


def optimize_parameters(generator: XAUUSDSignalGenerator, df: pd.DataFrame) -> Dict:
    """Very simple sweep over mode and SL/TP multipliers."""
    best = None
    results = []
    for mode in ['standard', 'aggressive']:
        for tp_mult in [0.75, 1.0, 1.25]:
            for sl_mult in [0.7, 0.8, 1.0]:
                gen = XAUUSDSignalGenerator(alpha_vantage_key=generator.alpha_vantage_key, mode=mode)
                gen.tp_pips = int(round(generator.tp_pips * tp_mult))
                gen.sl_pips = int(round(generator.sl_pips * sl_mult))
                bt = Backtester(gen, interval='5m')
                perf = bt.run(df)
                perf_row = {
                    'mode': mode,
                    'tp_pips': gen.tp_pips,
                    'sl_pips': gen.sl_pips,
                    'trades': perf['trades'],
                    'win_rate': perf['win_rate']
                }
                results.append(perf_row)
                if (best is None) or (perf['win_rate'] > best['win_rate']) or (perf['win_rate'] == best['win_rate'] and perf['trades'] > best['trades']):
                    best = perf_row
    return { 'best': best, 'results': results }
# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_single_signal():
    """Example: Generate single signal"""
    generator = XAUUSDSignalGenerator(alpha_vantage_key="ZGRKP8UJDN2ILXFY")
    
    # Get signal as dictionary
    signal = generator.generate_signal()
    print(json.dumps(signal, indent=2))
    
    # Or get as JSON string
    signal_json = generator.get_signal_json()
    print(signal_json)


def example_multiple_signals():
    """Example: Generate multiple signals"""
    generator = XAUUSDSignalGenerator(alpha_vantage_key="ZGRKP8UJDN2ILXFY")
    
    # Generate 5 signals
    signals = generator.generate_multiple_signals(count=5)
    print(json.dumps(signals, indent=2))
    
    # Or get as JSON string
    batch_json = generator.get_batch_signals_json(count=5)
    print(batch_json)


def example_performance_report():
    """Example: Generate performance report"""
    generator = XAUUSDSignalGenerator(alpha_vantage_key="ZGRKP8UJDN2ILXFY")
    
    # Simulate some signals with outcomes
    for i in range(10):
        signal = generator.generate_signal()
        # Simulate outcome (in real system, this comes from trade monitoring)
        generator.signal_history[-1]['outcome'] = 'win' if i < 8 else 'loss'
    
    # Get daily report
    daily_report = generator.calculate_performance_report(period="daily")
    print(json.dumps(daily_report, indent=2))
    
    # Get weekly report
    weekly_report = generator.calculate_performance_report(period="weekly")
    print(json.dumps(weekly_report, indent=2))


def example_api_endpoint():
    """Example: Use as API endpoint (Flask/FastAPI)"""
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    generator = XAUUSDSignalGenerator(alpha_vantage_key="ZGRKP8UJDN2ILXFY")
    
    @app.route('/api/signal', methods=['GET'])
    def get_signal():
        signal = generator.generate_signal()
        return jsonify(signal)
    
    @app.route('/api/signals/<int:count>', methods=['GET'])
    def get_signals(count):
        signals = generator.generate_multiple_signals(count=min(count, 10))
        return jsonify(signals)
    
    @app.route('/api/report/<period>', methods=['GET'])
    def get_report(period):
        report = generator.calculate_performance_report(period=period)
        return jsonify(report)
    
    # app.run(debug=True)  # Uncomment to run


# def main():
#     parser = argparse.ArgumentParser(description="XAUUSD Signal Generator")
#     parser.add_argument('--mode', default='standard', choices=['standard', 'aggressive'])
#     parser.add_argument('--action', default='signal', choices=['signal', 'backtest', 'optimize', 'notify'])
#     parser.add_argument('--interval', default='5m', choices=['1m','5m','15m','1h'])
#     parser.add_argument('--symbol', default='GC=F')
#     parser.add_argument('--telegram-token', default=None, help='Telegram bot token')
#     parser.add_argument('--telegram-chat-id', default=None, help='Telegram chat/channel id (e.g., -1001234567890)')
#     parser.add_argument('--supabase-url', default=None, help='Supabase project URL')
#     parser.add_argument('--supabase-key', default=None, help='Supabase anon or service key')
#     parser.add_argument('--start', default=None, help='ISO e.g. 2024-01-01T00:00:00Z')
#     parser.add_argument('--end', default=None, help='ISO e.g. 2024-03-01T00:00:00Z')
#     args = parser.parse_args()

#     generator = XAUUSDSignalGenerator(
#         alpha_vantage_key="demo",
#         mode=args.mode,
#         telegram_token=(args.telegram_token or "8281174730:AAHRTGP4ZRzloy9m0tNH_jND1kx-GcePNtE"),
#         telegram_chat_id=(args.telegram_chat_id or "-1002746890711")
#     )

#     if args.action == 'signal':
#         signal = generator.generate_signal()
#         print(json.dumps(signal, indent=2))
#         return

#     if args.action == 'notify':
#         signal = generator.generate_signal()
#         print(json.dumps(signal, indent=2))
#         return

#     # Backtest/optimize need historical data
#     if args.start and args.end:
#         start = datetime.fromisoformat(args.start.replace('Z','+00:00'))
#         end = datetime.fromisoformat(args.end.replace('Z','+00:00'))
#     else:
#         # default: last 14 days for 5m
#         end = datetime.now(UTC)
#         start = end - timedelta(days=14)

#     df = generator.fetch_market_data_range(args.symbol, args.interval, start, end)

#     if args.action == 'backtest':
#         bt = Backtester(generator, interval=args.interval)
#         perf = bt.run(df)
#         print(json.dumps(perf, indent=2))
#         return

#     if args.action == 'optimize':
#         result = optimize_parameters(generator, df)
#         print(json.dumps(result, indent=2))
#         return


def main():
    parser = argparse.ArgumentParser(description="XAUUSD Signal Generator")
    parser.add_argument('--mode', default='standard', choices=['standard', 'aggressive'])
    parser.add_argument('--action', default='signal', choices=['signal', 'backtest', 'optimize', 'notify', 'stats'])
    parser.add_argument('--interval', default='5m', choices=['1m','5m','15m','1h'])
    parser.add_argument('--symbol', default='GC=F')
    parser.add_argument('--telegram-token', default=None, help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', default=None, help='Telegram chat ID')
    parser.add_argument('--supabase-url', default=None, help='Supabase project URL')
    parser.add_argument('--supabase-key', default=None, help='Supabase anon/service key')
    parser.add_argument('--start', default=None, help='Start date for backtest')
    parser.add_argument('--end', default=None, help='End date for backtest')
    parser.add_argument('--stats-days', default=7, type=int, help='Days for stats calculation')
    args = parser.parse_args()

    # Initialize generator with all credentials
    generator = XAUUSDSignalGenerator(
        alpha_vantage_key="demo",
        mode=args.mode,
        telegram_token=(args.telegram_token or "8281174730:AAHRTGP4ZRzloy9m0tNH_jND1kx-GcePNtE"),
        telegram_chat_id=(args.telegram_chat_id or "-1002746890711"),
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key
    )

    # Generate signal and send to Telegram + save to Supabase
    if args.action == 'signal' or args.action == 'notify':
        signal = generator.generate_signal()
        print(json.dumps(signal, indent=2))
        
        # Print save status
        if signal.get('database_saved'):
            print(f"\n✅ Signal saved to Supabase (ID: {signal.get('database_id')})")
        if signal.get('telegram_sent'):
            print("✅ Signal sent to Telegram")
        return
    
    # Get performance stats from database
    if args.action == 'stats':
        stats = generator.get_performance_from_db(days=args.stats_days)
        print(json.dumps(stats, indent=2))
        
        # Also show recent signals
        recent = generator.get_recent_signals_from_db(limit=5)
        print(f"\n📊 Recent {len(recent)} signals:")
        for sig in recent:
            print(f"  - {sig.get('direction')} at {sig.get('entry_price')} "
                  f"(Confidence: {sig.get('confidence')}, "
                  f"Telegram: {'✅' if sig.get('telegram_sent') else '❌'})")
        return

    # Rest remains the same for backtest/optimize
    if args.start and args.end:
        start = datetime.fromisoformat(args.start.replace('Z','+00:00'))
        end = datetime.fromisoformat(args.end.replace('Z','+00:00'))
    else:
        end = datetime.now(UTC)
        start = end - timedelta(days=14)

    df = generator.fetch_market_data_range(args.symbol, args.interval, start, end)

    if args.action == 'backtest':
        bt = Backtester(generator, interval=args.interval)
        perf = bt.run(df)
        print(json.dumps(perf, indent=2))
        return

    if args.action == 'optimize':
        result = optimize_parameters(generator, df)
        print(json.dumps(result, indent=2))
        return


# Main execution
if __name__ == "__main__":
    main()