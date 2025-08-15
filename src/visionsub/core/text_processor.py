"""
Text processing utilities for OCR results
"""
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing and cleanup utilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get("language", "zh")
        
    def post_process_text(self, text: str) -> str:
        """Post-process OCR text to improve quality"""
        if not text:
            return text
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Basic punctuation cleanup
        text = self._cleanup_punctuation(text)
        
        # Language-specific processing
        if self.language == "zh":
            text = self._process_chinese(text)
        elif self.language == "en":
            text = self._process_english(text)
        
        return text.strip()
    
    def _cleanup_punctuation(self, text: str) -> str:
        """Clean up punctuation"""
        # Fix common OCR punctuation errors
        text = re.sub(r'\s+([.,;:!?"\'])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?"\'])\s+', r'\1 ', text)  # Ensure space after punctuation
        text = re.sub(r'([.,;:!?"\']){2,}', r'\1', text)  # Remove duplicate punctuation
        
        return text
    
    def _process_chinese(self, text: str) -> str:
        """Process Chinese text specifically"""
        # Remove common Chinese OCR errors
        text = re.sub(r'[，。、；：""''！？（）【】《》]', lambda m: {
            '，': ',', '。': '.', '、': ',', '；': ';', '：': ':',
            '""': '"', "''": "'", '！': '!', '？': '?',
            '（': '(', '）': ')', '【': '[', '】': ']',
            '《': '<', '》': '>'
        }.get(m.group(0), m.group(0)), text)
        
        # Fix common character substitutions
        text = re.sub(r'０', '0', text)
        text = re.sub(r'１', '1', text)
        text = re.sub(r'２', '2', text)
        text = re.sub(r'３', '3', text)
        text = re.sub(r'４', '4', text)
        text = re.sub(r'５', '5', text)
        text = re.sub(r'６', '6', text)
        text = re.sub(r'７', '7', text)
        text = re.sub(r'８', '8', text)
        text = re.sub(r'９', '9', text)
        
        return text
    
    def _process_english(self, text: str) -> str:
        """Process English text specifically"""
        # Fix common English OCR errors
        text = re.sub(r'I\s+', 'I ', text)  # Fix spacing after "I"
        text = re.sub(r'\s+I\s', ' I ', text)  # Fix spacing around "I"
        
        # Fix common character substitutions
        text = re.sub(r'0', 'O', text)  # Common confusion between 0 and O
        text = re.sub(r'1', 'I', text)  # Common confusion between 1 and I
        text = re.sub(r'5', 'S', text)  # Common confusion between 5 and S
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 2:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word-based similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def update_config(self, config: Dict[str, Any]):
        """Update processor configuration"""
        self.config.update(config)
        self.language = self.config.get("language", "zh")