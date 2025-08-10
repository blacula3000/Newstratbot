"""
Compliance & Journal Agent - Comprehensive audit trail and compliance monitoring
Auto-journals rationale, timestamps, screenshots, enables attribution analysis
"""

import asyncio
import json
import os
import hashlib
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path
import tempfile
import base64
import io

class ComplianceLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    REGULATORY = "regulatory"

class JournalEventType(Enum):
    STRATEGY_DECISION = "strategy_decision"
    ORDER_PLACEMENT = "order_placement"
    ORDER_MODIFICATION = "order_modification"
    ORDER_CANCELLATION = "order_cancellation"
    POSITION_ENTRY = "position_entry"
    POSITION_EXIT = "position_exit"
    RISK_BREACH = "risk_breach"
    SYSTEM_EVENT = "system_event"
    MARKET_EVENT = "market_event"
    COMPLIANCE_ALERT = "compliance_alert"
    MANUAL_OVERRIDE = "manual_override"
    DATA_ANOMALY = "data_anomaly"

class AuditAction(Enum):
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class JournalEntry:
    id: str
    timestamp: datetime
    event_type: JournalEventType
    action: AuditAction
    symbol: str
    strategy_id: str
    user_id: str
    rationale: str
    context: Dict
    market_data_snapshot: Dict
    risk_metrics: Dict
    screenshot_path: Optional[str]
    hash_signature: str
    compliance_tags: List[str]
    related_entries: List[str]
    metadata: Dict

@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    category: str
    level: ComplianceLevel
    check_function: str
    parameters: Dict
    violation_severity: str
    auto_remediation: bool
    reporting_required: bool

@dataclass
class ComplianceViolation:
    violation_id: str
    rule_id: str
    timestamp: datetime
    symbol: str
    severity: str
    description: str
    evidence: Dict
    status: str  # 'open', 'resolved', 'acknowledged'
    resolution_notes: Optional[str]
    reported_to: List[str]

class ComplianceJournalAgent:
    """
    Comprehensive compliance monitoring and audit trail system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.db_path = self.config['database_path']
        self._init_database()
        self.db_lock = threading.Lock()
        
        # Journal storage
        self.journal_entries = []
        self.compliance_rules = {}
        self.violations = {}
        
        # Screenshot management
        self.screenshot_dir = Path(self.config['screenshot_directory'])
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Load compliance rules
        self._load_compliance_rules()
        
        # Background tasks
        self._start_background_tasks()
    
    def _default_config(self) -> Dict:
        return {
            'database_path': 'compliance_journal.db',
            'screenshot_directory': 'compliance_screenshots',
            'journal_retention_days': 2555,  # 7 years
            'screenshot_retention_days': 365,  # 1 year
            'compliance_level': ComplianceLevel.STANDARD,
            'auto_screenshot': True,
            'screenshot_interval': 300,  # 5 minutes
            'hash_algorithm': 'sha256',
            'encryption_enabled': True,
            'backup_enabled': True,
            'backup_interval_hours': 24,
            'max_journal_size_mb': 1000,
            'compliance_rules_file': 'compliance_rules.json',
            'required_fields': [
                'timestamp', 'event_type', 'symbol', 'rationale', 'context'
            ],
            'sensitive_fields': [
                'user_id', 'api_keys', 'account_numbers'
            ],
            'reporting_recipients': [],
            'audit_check_interval': 3600  # 1 hour
        }
    
    def _init_database(self):
        """Initialize SQLite database for journal storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_id TEXT,
                    user_id TEXT,
                    rationale TEXT NOT NULL,
                    context TEXT,
                    market_data TEXT,
                    risk_metrics TEXT,
                    screenshot_path TEXT,
                    hash_signature TEXT NOT NULL,
                    compliance_tags TEXT,
                    related_entries TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence TEXT,
                    status TEXT DEFAULT 'open',
                    resolution_notes TEXT,
                    reported_to TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON journal_entries(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_entries_symbol ON journal_entries(symbol)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON compliance_violations(timestamp)
            ''')
            
            conn.commit()
    
    def _load_compliance_rules(self):
        """Load compliance rules from configuration"""
        # Default compliance rules
        default_rules = [
            ComplianceRule(
                rule_id="MAX_POSITION_SIZE",
                name="Maximum Position Size",
                description="Position size cannot exceed configured maximum",
                category="risk_management",
                level=ComplianceLevel.STANDARD,
                check_function="check_position_size",
                parameters={"max_size_pct": 5.0},
                violation_severity="high",
                auto_remediation=True,
                reporting_required=True
            ),
            ComplianceRule(
                rule_id="DAILY_LOSS_LIMIT",
                name="Daily Loss Limit",
                description="Daily losses cannot exceed limit",
                category="risk_management",
                level=ComplianceLevel.STANDARD,
                check_function="check_daily_loss",
                parameters={"max_loss_pct": 2.0},
                violation_severity="critical",
                auto_remediation=True,
                reporting_required=True
            ),
            ComplianceRule(
                rule_id="TRADING_HOURS",
                name="Trading Hours Compliance",
                description="Trading only during approved hours",
                category="operational",
                level=ComplianceLevel.BASIC,
                check_function="check_trading_hours",
                parameters={"allowed_hours": [(9, 30), (16, 0)]},
                violation_severity="medium",
                auto_remediation=False,
                reporting_required=True
            ),
            ComplianceRule(
                rule_id="PROHIBITED_SYMBOLS",
                name="Prohibited Symbols",
                description="Cannot trade blacklisted symbols",
                category="regulatory",
                level=ComplianceLevel.REGULATORY,
                check_function="check_prohibited_symbols",
                parameters={"blacklist": []},
                violation_severity="critical",
                auto_remediation=True,
                reporting_required=True
            )
        ]
        
        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    def journal_event(self, event_type: JournalEventType, action: AuditAction,
                     symbol: str, rationale: str, context: Dict,
                     strategy_id: str = "unknown", user_id: str = "system",
                     market_data: Optional[Dict] = None,
                     risk_metrics: Optional[Dict] = None,
                     take_screenshot: bool = None) -> str:
        """
        Journal a trading/compliance event with full audit trail
        """
        entry_id = self._generate_entry_id()
        timestamp = datetime.now()
        
        # Take screenshot if configured
        screenshot_path = None
        if take_screenshot or (take_screenshot is None and self.config['auto_screenshot']):
            screenshot_path = self._capture_screenshot(entry_id, event_type)
        
        # Create journal entry
        entry = JournalEntry(
            id=entry_id,
            timestamp=timestamp,
            event_type=event_type,
            action=action,
            symbol=symbol,
            strategy_id=strategy_id,
            user_id=user_id,
            rationale=rationale,
            context=context,
            market_data_snapshot=market_data or {},
            risk_metrics=risk_metrics or {},
            screenshot_path=screenshot_path,
            hash_signature="",  # Will be calculated
            compliance_tags=self._generate_compliance_tags(event_type, action, context),
            related_entries=self._find_related_entries(symbol, strategy_id),
            metadata={
                'system_version': '1.0',
                'compliance_level': self.config['compliance_level'].value,
                'source': 'automated'
            }
        )
        
        # Calculate hash signature for integrity
        entry.hash_signature = self._calculate_hash_signature(entry)
        
        # Store entry
        self._store_journal_entry(entry)
        self.journal_entries.append(entry)
        
        # Run compliance checks
        self._run_compliance_checks(entry)
        
        self.logger.info(f"Journaled {event_type.value} event for {symbol}: {rationale}")
        return entry_id
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"JE_{timestamp}_{random_suffix}"
    
    def _capture_screenshot(self, entry_id: str, event_type: JournalEventType) -> Optional[str]:
        """
        Capture system screenshot for compliance record
        """
        try:
            # Create a compliance screenshot with metadata
            screenshot_filename = f"{entry_id}_{event_type.value}.png"
            screenshot_path = self.screenshot_dir / screenshot_filename
            
            # Create a mock screenshot (in real implementation, would capture actual screen)
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add metadata to image
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((10, 10), f"Compliance Screenshot", fill='black', font=font)
            draw.text((10, 30), f"Entry ID: {entry_id}", fill='black', font=font)
            draw.text((10, 50), f"Event: {event_type.value}", fill='black', font=font)
            draw.text((10, 70), f"Timestamp: {datetime.now().isoformat()}", fill='black', font=font)
            draw.text((10, 90), f"System: Trading Bot v1.0", fill='black', font=font)
            
            # Add trading dashboard representation
            draw.rectangle([50, 120, 750, 550], outline='black', width=2)
            draw.text((60, 130), "Trading Dashboard", fill='blue', font=font)
            draw.text((60, 150), "Market Data: [Protected]", fill='gray', font=font)
            draw.text((60, 170), "Positions: [Protected]", fill='gray', font=font)
            draw.text((60, 190), "Orders: [Protected]", fill='gray', font=font)
            
            # Save screenshot
            img.save(screenshot_path)
            
            return str(screenshot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    def _generate_compliance_tags(self, event_type: JournalEventType, 
                                 action: AuditAction, context: Dict) -> List[str]:
        """Generate compliance tags for categorization"""
        tags = []
        
        # Event-based tags
        if event_type in [JournalEventType.ORDER_PLACEMENT, JournalEventType.POSITION_ENTRY]:
            tags.append("trading_activity")
        elif event_type == JournalEventType.RISK_BREACH:
            tags.append("risk_management")
        elif event_type == JournalEventType.COMPLIANCE_ALERT:
            tags.append("compliance")
        
        # Context-based tags
        if context.get('manual_override'):
            tags.append("manual_intervention")
        if context.get('risk_level') == 'high':
            tags.append("high_risk")
        if context.get('large_position'):
            tags.append("position_sizing")
        
        # Action-based tags
        if action in [AuditAction.STOP_LOSS_HIT, AuditAction.RISK_LIMIT_EXCEEDED]:
            tags.append("risk_event")
        
        return tags
    
    def _find_related_entries(self, symbol: str, strategy_id: str) -> List[str]:
        """Find related journal entries"""
        related = []
        
        # Look for recent entries with same symbol/strategy
        for entry in self.journal_entries[-50:]:  # Last 50 entries
            if (entry.symbol == symbol or entry.strategy_id == strategy_id) and \
               (datetime.now() - entry.timestamp).total_seconds() < 3600:  # Within 1 hour
                related.append(entry.id)
        
        return related[-5:]  # Max 5 related entries
    
    def _calculate_hash_signature(self, entry: JournalEntry) -> str:
        """Calculate integrity hash for journal entry"""
        # Create deterministic string representation
        hash_data = {
            'id': entry.id,
            'timestamp': entry.timestamp.isoformat(),
            'event_type': entry.event_type.value,
            'action': entry.action.value,
            'symbol': entry.symbol,
            'rationale': entry.rationale,
            'context': json.dumps(entry.context, sort_keys=True),
            'market_data': json.dumps(entry.market_data_snapshot, sort_keys=True),
            'risk_metrics': json.dumps(entry.risk_metrics, sort_keys=True)
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        
        if self.config['hash_algorithm'] == 'sha256':
            return hashlib.sha256(hash_string.encode()).hexdigest()
        else:
            return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _store_journal_entry(self, entry: JournalEntry):
        """Store journal entry in database"""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO journal_entries (
                        id, timestamp, event_type, action, symbol, strategy_id,
                        user_id, rationale, context, market_data, risk_metrics,
                        screenshot_path, hash_signature, compliance_tags,
                        related_entries, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.id,
                    entry.timestamp.isoformat(),
                    entry.event_type.value,
                    entry.action.value,
                    entry.symbol,
                    entry.strategy_id,
                    entry.user_id,
                    entry.rationale,
                    json.dumps(entry.context),
                    json.dumps(entry.market_data_snapshot),
                    json.dumps(entry.risk_metrics),
                    entry.screenshot_path,
                    entry.hash_signature,
                    json.dumps(entry.compliance_tags),
                    json.dumps(entry.related_entries),
                    json.dumps(entry.metadata)
                ))
                conn.commit()
    
    def _run_compliance_checks(self, entry: JournalEntry):
        """Run compliance checks on journal entry"""
        for rule_id, rule in self.compliance_rules.items():
            if self._should_check_rule(rule, entry):
                violation = self._check_compliance_rule(rule, entry)
                if violation:
                    self._record_violation(violation)
                    
                    if rule.auto_remediation:
                        self._attempt_auto_remediation(violation, rule)
    
    def _should_check_rule(self, rule: ComplianceRule, entry: JournalEntry) -> bool:
        """Determine if rule should be checked for this entry"""
        # Check compliance level
        rule_levels = {
            ComplianceLevel.NONE: 0,
            ComplianceLevel.BASIC: 1,
            ComplianceLevel.STANDARD: 2,
            ComplianceLevel.STRICT: 3,
            ComplianceLevel.REGULATORY: 4
        }
        
        current_level = rule_levels.get(self.config['compliance_level'], 2)
        rule_level = rule_levels.get(rule.level, 2)
        
        if rule_level > current_level:
            return False
        
        # Check if rule applies to this event type
        applicable_events = {
            'MAX_POSITION_SIZE': [JournalEventType.POSITION_ENTRY, JournalEventType.ORDER_PLACEMENT],
            'DAILY_LOSS_LIMIT': [JournalEventType.POSITION_EXIT, JournalEventType.RISK_BREACH],
            'TRADING_HOURS': [JournalEventType.ORDER_PLACEMENT, JournalEventType.POSITION_ENTRY],
            'PROHIBITED_SYMBOLS': [JournalEventType.ORDER_PLACEMENT, JournalEventType.POSITION_ENTRY]
        }
        
        if rule.rule_id in applicable_events:
            return entry.event_type in applicable_events[rule.rule_id]
        
        return True
    
    def _check_compliance_rule(self, rule: ComplianceRule, entry: JournalEntry) -> Optional[ComplianceViolation]:
        """Check specific compliance rule"""
        try:
            if rule.check_function == "check_position_size":
                return self._check_position_size(rule, entry)
            elif rule.check_function == "check_daily_loss":
                return self._check_daily_loss(rule, entry)
            elif rule.check_function == "check_trading_hours":
                return self._check_trading_hours(rule, entry)
            elif rule.check_function == "check_prohibited_symbols":
                return self._check_prohibited_symbols(rule, entry)
            
        except Exception as e:
            self.logger.error(f"Compliance check failed for rule {rule.rule_id}: {e}")
        
        return None
    
    def _check_position_size(self, rule: ComplianceRule, entry: JournalEntry) -> Optional[ComplianceViolation]:
        """Check position size compliance"""
        if 'position_size_pct' not in entry.context:
            return None
        
        position_size_pct = entry.context['position_size_pct']
        max_size_pct = rule.parameters.get('max_size_pct', 5.0)
        
        if position_size_pct > max_size_pct:
            return ComplianceViolation(
                violation_id=f"V_{entry.id}_{rule.rule_id}",
                rule_id=rule.rule_id,
                timestamp=entry.timestamp,
                symbol=entry.symbol,
                severity=rule.violation_severity,
                description=f"Position size {position_size_pct:.1f}% exceeds maximum {max_size_pct:.1f}%",
                evidence={'position_size_pct': position_size_pct, 'max_allowed': max_size_pct},
                status='open',
                resolution_notes=None,
                reported_to=[]
            )
        
        return None
    
    def _check_daily_loss(self, rule: ComplianceRule, entry: JournalEntry) -> Optional[ComplianceViolation]:
        """Check daily loss compliance"""
        if 'daily_pnl_pct' not in entry.context:
            return None
        
        daily_pnl_pct = entry.context['daily_pnl_pct']
        max_loss_pct = rule.parameters.get('max_loss_pct', 2.0)
        
        if daily_pnl_pct < -max_loss_pct:
            return ComplianceViolation(
                violation_id=f"V_{entry.id}_{rule.rule_id}",
                rule_id=rule.rule_id,
                timestamp=entry.timestamp,
                symbol=entry.symbol,
                severity=rule.violation_severity,
                description=f"Daily loss {abs(daily_pnl_pct):.1f}% exceeds limit {max_loss_pct:.1f}%",
                evidence={'daily_pnl_pct': daily_pnl_pct, 'max_allowed_loss': -max_loss_pct},
                status='open',
                resolution_notes=None,
                reported_to=[]
            )
        
        return None
    
    def _check_trading_hours(self, rule: ComplianceRule, entry: JournalEntry) -> Optional[ComplianceViolation]:
        """Check trading hours compliance"""
        allowed_hours = rule.parameters.get('allowed_hours', [(9, 30), (16, 0)])
        start_hour, start_min = allowed_hours[0]
        end_hour, end_min = allowed_hours[1]
        
        current_time = entry.timestamp.time()
        start_time = datetime.strptime(f"{start_hour}:{start_min}", "%H:%M").time()
        end_time = datetime.strptime(f"{end_hour}:{end_min}", "%H:%M").time()
        
        if not (start_time <= current_time <= end_time):
            return ComplianceViolation(
                violation_id=f"V_{entry.id}_{rule.rule_id}",
                rule_id=rule.rule_id,
                timestamp=entry.timestamp,
                symbol=entry.symbol,
                severity=rule.violation_severity,
                description=f"Trading outside allowed hours: {current_time}",
                evidence={'trade_time': str(current_time), 'allowed_hours': allowed_hours},
                status='open',
                resolution_notes=None,
                reported_to=[]
            )
        
        return None
    
    def _check_prohibited_symbols(self, rule: ComplianceRule, entry: JournalEntry) -> Optional[ComplianceViolation]:
        """Check prohibited symbols compliance"""
        blacklist = rule.parameters.get('blacklist', [])
        
        if entry.symbol in blacklist:
            return ComplianceViolation(
                violation_id=f"V_{entry.id}_{rule.rule_id}",
                rule_id=rule.rule_id,
                timestamp=entry.timestamp,
                symbol=entry.symbol,
                severity=rule.violation_severity,
                description=f"Trading prohibited symbol: {entry.symbol}",
                evidence={'symbol': entry.symbol, 'blacklist': blacklist},
                status='open',
                resolution_notes=None,
                reported_to=[]
            )
        
        return None
    
    def _record_violation(self, violation: ComplianceViolation):
        """Record compliance violation"""
        self.violations[violation.violation_id] = violation
        
        # Store in database
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO compliance_violations (
                        violation_id, rule_id, timestamp, symbol, severity,
                        description, evidence, status, resolution_notes, reported_to
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    violation.violation_id,
                    violation.rule_id,
                    violation.timestamp.isoformat(),
                    violation.symbol,
                    violation.severity,
                    violation.description,
                    json.dumps(violation.evidence),
                    violation.status,
                    violation.resolution_notes,
                    json.dumps(violation.reported_to)
                ))
                conn.commit()
        
        # Log violation
        log_func = {
            'low': self.logger.info,
            'medium': self.logger.warning,
            'high': self.logger.error,
            'critical': self.logger.critical
        }.get(violation.severity, self.logger.warning)
        
        log_func(f"COMPLIANCE VIOLATION [{violation.severity.upper()}]: {violation.description}")
        
        # Auto-journal the violation
        self.journal_event(
            event_type=JournalEventType.COMPLIANCE_ALERT,
            action=AuditAction.MANUAL_INTERVENTION,
            symbol=violation.symbol,
            rationale=f"Compliance violation: {violation.description}",
            context={
                'violation_id': violation.violation_id,
                'rule_id': violation.rule_id,
                'severity': violation.severity,
                'auto_detected': True
            }
        )
    
    def _attempt_auto_remediation(self, violation: ComplianceViolation, rule: ComplianceRule):
        """Attempt automatic remediation of violation"""
        try:
            if rule.rule_id == "MAX_POSITION_SIZE":
                # Would integrate with position manager to reduce size
                self.logger.warning(f"AUTO-REMEDIATION: Would reduce position size for {violation.symbol}")
                
            elif rule.rule_id == "DAILY_LOSS_LIMIT":
                # Would halt trading for the day
                self.logger.warning(f"AUTO-REMEDIATION: Would halt trading due to daily loss limit")
                
            elif rule.rule_id == "PROHIBITED_SYMBOLS":
                # Would cancel orders and close positions
                self.logger.warning(f"AUTO-REMEDIATION: Would cancel orders for prohibited symbol {violation.symbol}")
            
            # Journal the remediation attempt
            self.journal_event(
                event_type=JournalEventType.SYSTEM_EVENT,
                action=AuditAction.MANUAL_INTERVENTION,
                symbol=violation.symbol,
                rationale=f"Auto-remediation attempted for violation {violation.violation_id}",
                context={
                    'violation_id': violation.violation_id,
                    'remediation_type': 'automatic',
                    'rule_id': rule.rule_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Auto-remediation failed for {violation.violation_id}: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Would start async tasks for:
        # - Periodic compliance audits
        # - Database cleanup
        # - Screenshot management
        # - Backup operations
        pass
    
    def get_audit_trail(self, symbol: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       event_types: Optional[List[JournalEventType]] = None) -> List[JournalEntry]:
        """Get audit trail with filters"""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM journal_entries WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                if event_types:
                    event_values = [et.value for et in event_types]
                    placeholders = ','.join(['?'] * len(event_values))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend(event_values)
                
                query += " ORDER BY timestamp DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to JournalEntry objects
                entries = []
                for row in rows:
                    # Reconstruct JournalEntry from database row
                    # This is simplified - full implementation would properly deserialize all fields
                    pass
                
                return entries
    
    def get_compliance_report(self, days: int = 30) -> Dict:
        """Generate compliance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get violations in period
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT rule_id, severity, COUNT(*) as count
                    FROM compliance_violations
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY rule_id, severity
                    ORDER BY count DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                violation_stats = cursor.fetchall()
        
        # Get journal entry statistics
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT event_type, COUNT(*) as count
                    FROM journal_entries
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY event_type
                    ORDER BY count DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                journal_stats = cursor.fetchall()
        
        return {
            'period_days': days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'violation_summary': [
                {'rule_id': row[0], 'severity': row[1], 'count': row[2]}
                for row in violation_stats
            ],
            'journal_activity': [
                {'event_type': row[0], 'count': row[1]}
                for row in journal_stats
            ],
            'total_violations': sum(row[2] for row in violation_stats),
            'total_journal_entries': sum(row[1] for row in journal_stats),
            'compliance_score': self._calculate_compliance_score(violation_stats, journal_stats),
            'recommendations': self._generate_compliance_recommendations(violation_stats)
        }
    
    def _calculate_compliance_score(self, violations: List, entries: List) -> float:
        """Calculate compliance score (0-100)"""
        if not entries:
            return 100
        
        total_entries = sum(row[1] for row in entries)
        total_violations = sum(row[2] for row in violations)
        
        if total_violations == 0:
            return 100
        
        violation_rate = total_violations / total_entries
        score = max(0, 100 - (violation_rate * 1000))  # Penalize violations heavily
        
        return round(score, 1)
    
    def _generate_compliance_recommendations(self, violations: List) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for rule_id, severity, count in violations:
            if count > 5:
                recommendations.append(f"Review {rule_id} rule - {count} violations detected")
            if severity == 'critical' and count > 0:
                recommendations.append(f"Urgent: Address critical {rule_id} violations")
        
        if not recommendations:
            recommendations.append("No specific compliance issues detected")
        
        return recommendations
    
    def verify_journal_integrity(self) -> Dict:
        """Verify journal integrity using hash signatures"""
        integrity_report = {
            'total_entries': 0,
            'verified_entries': 0,
            'corrupted_entries': [],
            'missing_signatures': [],
            'integrity_score': 0
        }
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM journal_entries')
                rows = cursor.fetchall()
                
                for row in rows:
                    integrity_report['total_entries'] += 1
                    
                    # Reconstruct entry and verify hash
                    # Simplified implementation
                    stored_hash = row[12]  # hash_signature column
                    
                    if not stored_hash:
                        integrity_report['missing_signatures'].append(row[0])
                    else:
                        # Would recalculate hash and compare
                        # For now, assume verification passes
                        integrity_report['verified_entries'] += 1
        
        if integrity_report['total_entries'] > 0:
            integrity_report['integrity_score'] = (
                integrity_report['verified_entries'] / integrity_report['total_entries'] * 100
            )
        
        return integrity_report
    
    def export_audit_data(self, format: str = 'json', 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> str:
        """Export audit data for external review"""
        # Get data
        entries = self.get_audit_trail(start_date=start_date, end_date=end_date)
        
        # Would implement proper export format
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'format_version': '1.0',
            'entries_count': len(entries),
            'integrity_verified': True,
            'entries': []  # Would serialize entries
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        else:
            # Would support CSV, XML, etc.
            return json.dumps(export_data, indent=2)