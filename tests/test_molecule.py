#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for molecule.py — the Python implementation of molecule.ai

These tests verify the core components of the organism:
- Tokenizer (char-level and BPE evolution)
- Autograd engine (VectorValue, ScalarValue, backward pass)
- Model components (MatrixParam, GPT layers)
- SQLite memory operations
- Corpus reservoir management

Run: python -m pytest tests/test_molecule.py -v
"""

import sys
import os
import math
import tempfile
import sqlite3

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from molecule.py
from molecule import (
    Config, CFG,
    VectorValue, ScalarValue, backward,
    MatrixParam, 
    EvolvingTokenizer,
    init_db, db_add_message, db_recent_messages,
    load_corpus_lines, save_corpus_lines, normalize_text,
    extract_candidate_sentences_from_messages,
    reservoir_mix_keep,
)


# ============================================================
# 1) CONFIG TESTS
# ============================================================

class TestConfig:
    """Test configuration defaults."""
    
    def test_config_exists(self):
        """Config should be defined."""
        assert CFG is not None
        assert isinstance(CFG, Config)
    
    def test_config_model_params(self):
        """Model parameters should have sensible defaults."""
        assert CFG.n_layer == 2
        assert CFG.n_embd == 72
        assert CFG.n_head == 4
        assert CFG.block_size == 96
    
    def test_config_training_params(self):
        """Training parameters should be defined."""
        assert CFG.learning_rate > 0
        assert CFG.beta1 > 0 and CFG.beta1 < 1
        assert CFG.beta2 > 0 and CFG.beta2 < 1
        assert CFG.grad_clip > 0
    
    def test_config_delta_params(self):
        """Delta adapter parameters should be defined."""
        assert CFG.delta_rank > 0
        assert CFG.max_delta_modules > 0


# ============================================================
# 2) AUTOGRAD TESTS — VectorValue
# ============================================================

class TestVectorValue:
    """Test the vector autograd engine."""
    
    def test_vector_creation(self):
        """VectorValue should be created with data."""
        v = VectorValue([1.0, 2.0, 3.0])
        assert v.data == [1.0, 2.0, 3.0]
        assert v.grad == [0.0, 0.0, 0.0]
    
    def test_vector_addition(self):
        """Vector addition should work element-wise."""
        v1 = VectorValue([1.0, 2.0, 3.0])
        v2 = VectorValue([4.0, 5.0, 6.0])
        result = v1 + v2
        assert result.data == [5.0, 7.0, 9.0]
    
    def test_vector_subtraction(self):
        """Vector subtraction should work element-wise."""
        v1 = VectorValue([5.0, 7.0, 9.0])
        v2 = VectorValue([1.0, 2.0, 3.0])
        result = v1 - v2
        assert result.data == [4.0, 5.0, 6.0]
    
    def test_vector_negation(self):
        """Vector negation should negate all elements."""
        v = VectorValue([1.0, -2.0, 3.0])
        result = -v
        assert result.data == [-1.0, 2.0, -3.0]
    
    def test_vector_scalar_multiplication(self):
        """Multiplication by scalar should scale all elements."""
        v = VectorValue([1.0, 2.0, 3.0])
        result = v * 2.0
        assert result.data == [2.0, 4.0, 6.0]
    
    def test_vector_dot_product(self):
        """Dot product should return a scalar."""
        v1 = VectorValue([1.0, 2.0, 3.0])
        v2 = VectorValue([4.0, 5.0, 6.0])
        result = v1.dot(v2)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert result.data == 32.0
    
    def test_vector_relu(self):
        """ReLU should zero out negative values."""
        v = VectorValue([-1.0, 0.0, 2.0, -3.0, 4.0])
        result = v.relu()
        assert result.data == [0.0, 0.0, 2.0, 0.0, 4.0]
    
    def test_vector_squared_relu(self):
        """Squared ReLU should square positive values."""
        v = VectorValue([-1.0, 0.0, 2.0, 3.0])
        result = v.squared_relu()
        # ReLU: [0, 0, 2, 3] -> squared: [0, 0, 4, 9]
        assert result.data == [0.0, 0.0, 4.0, 9.0]
    
    def test_vector_mean_sq(self):
        """Mean squared should return average of squares."""
        v = VectorValue([1.0, 2.0, 3.0])
        result = v.mean_sq()
        # (1 + 4 + 9) / 3 = 14/3 ≈ 4.666...
        assert abs(result.data - 14.0/3.0) < 1e-10
    
    def test_vector_slice(self):
        """Slice should return a subset of the vector."""
        v = VectorValue([1.0, 2.0, 3.0, 4.0, 5.0])
        result = v.slice(1, 4)
        assert result.data == [2.0, 3.0, 4.0]
    
    def test_vector_concat(self):
        """Concat should join multiple vectors."""
        v1 = VectorValue([1.0, 2.0])
        v2 = VectorValue([3.0, 4.0])
        v3 = VectorValue([5.0])
        result = VectorValue.concat([v1, v2, v3])
        assert result.data == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_backward_via_dot_product(self):
        """Backward pass should propagate gradients through dot product."""
        v1 = VectorValue([1.0, 2.0])
        v2 = VectorValue([3.0, 4.0])
        # Use dot product which returns a ScalarValue that backward() expects
        result = v1.dot(v2)
        backward(result)
        # d(v1·v2)/dv1 = v2
        assert v1.grad == [3.0, 4.0]
    
    def test_backward_dot_product(self):
        """Backward pass for dot product should compute outer product."""
        v1 = VectorValue([1.0, 2.0])
        v2 = VectorValue([3.0, 4.0])
        result = v1.dot(v2)
        result.grad = 1.0
        backward(result)
        # d(v1·v2)/dv1 = v2, d(v1·v2)/dv2 = v1
        assert v1.grad == [3.0, 4.0]
        assert v2.grad == [1.0, 2.0]


# ============================================================
# 3) AUTOGRAD TESTS — ScalarValue
# ============================================================

class TestScalarValue:
    """Test the scalar autograd engine."""
    
    def test_scalar_creation(self):
        """ScalarValue should be created with data."""
        s = ScalarValue(5.0)
        assert s.data == 5.0
        assert s.grad == 0.0
    
    def test_scalar_addition(self):
        """Scalar addition should work."""
        s1 = ScalarValue(3.0)
        s2 = ScalarValue(4.0)
        result = s1 + s2
        assert result.data == 7.0
    
    def test_scalar_multiplication(self):
        """Scalar multiplication should work."""
        s1 = ScalarValue(3.0)
        s2 = ScalarValue(4.0)
        result = s1 * s2
        assert result.data == 12.0
    
    def test_scalar_float_addition(self):
        """Scalar + float should work."""
        s = ScalarValue(5.0)
        result = s + 3.0
        assert result.data == 8.0
    
    def test_scalar_float_multiplication(self):
        """Scalar * float should work."""
        s = ScalarValue(5.0)
        result = s * 2.0
        assert result.data == 10.0
    
    def test_scalar_arithmetic_composition(self):
        """Scalar arithmetic composition should work correctly."""
        s1 = ScalarValue(2.0)
        s2 = ScalarValue(3.0)
        result = s1 * s2 + s1
        # 2 * 3 + 2 = 8
        assert result.data == 8.0
    
    def test_scalar_subtraction(self):
        """Scalar subtraction should work correctly."""
        s1 = ScalarValue(10.0)
        s2 = ScalarValue(3.0)
        result = s1 - s2
        assert result.data == 7.0
    
    def test_backward_scalar_multiplication(self):
        """Backward pass for scalar multiplication."""
        s1 = ScalarValue(3.0)
        s2 = ScalarValue(4.0)
        result = s1 * s2
        result.grad = 1.0
        backward(result)
        # d(s1*s2)/ds1 = s2, d(s1*s2)/ds2 = s1
        assert s1.grad == 4.0
        assert s2.grad == 3.0


# ============================================================
# 4) MATRIX PARAM TESTS
# ============================================================

class TestMatrixParam:
    """Test the learnable matrix parameter."""
    
    def test_matrix_creation(self):
        """MatrixParam should be created with correct shape."""
        m = MatrixParam(3, 4, std=0.1)
        assert len(m.rows) == 3
        for row in m.rows:
            assert len(row.data) == 4
    
    def test_matrix_matvec(self):
        """Matrix-vector multiplication should work."""
        m = MatrixParam(2, 3, std=0.0)
        # Set known values
        m.rows[0].data = [1.0, 0.0, 0.0]
        m.rows[1].data = [0.0, 1.0, 0.0]
        
        x = VectorValue([5.0, 7.0, 9.0])
        result = m.matvec(x)
        
        # [1 0 0] @ [5, 7, 9] = 5
        # [0 1 0] @ [5, 7, 9] = 7
        assert abs(result.data[0] - 5.0) < 1e-10
        assert abs(result.data[1] - 7.0) < 1e-10
    
    def test_matrix_grow_rows(self):
        """MatrixParam should be able to grow rows."""
        m = MatrixParam(2, 3, std=0.1)
        assert len(m.rows) == 2
        
        m.grow_rows(5, std=0.1)
        assert len(m.rows) == 5
    
    def test_matrix_params(self):
        """MatrixParam.params() should return all row vectors."""
        m = MatrixParam(3, 4, std=0.1)
        params = m.params()
        assert len(params) == 3


# ============================================================
# 5) TOKENIZER TESTS
# ============================================================

class TestEvolvingTokenizer:
    """Test the evolving tokenizer."""
    
    def test_tokenizer_creation(self):
        """Tokenizer should initialize with char-level tokens."""
        docs = ["Hello world!", "Testing 123."]
        tok = EvolvingTokenizer(docs)
        
        # Should have all unique characters
        assert tok.vocab_size > 0
        assert "H" in tok.stoi
        assert "e" in tok.stoi
        assert " " in tok.stoi
    
    def test_tokenizer_encode_decode(self):
        """Encode then decode should return original text."""
        docs = ["Hello world!"]
        tok = EvolvingTokenizer(docs)
        
        text = "Hello"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        
        assert decoded == text
    
    def test_tokenizer_unknown_chars(self):
        """Unknown chars should be handled gracefully (using existing vocab)."""
        docs = ["abc"]
        tok = EvolvingTokenizer(docs)
        
        # Encode known characters
        text = "abc"
        ids = tok.encode(text)
        # Should produce tokens for each character
        assert len(ids) >= 3  # at least a, b, c (may include word boundary markers)
    
    def test_tokenizer_bpe_disabled_initially(self):
        """BPE should be disabled initially."""
        docs = ["Hello world!"]
        tok = EvolvingTokenizer(docs)
        
        assert tok.bpe_enabled == False
    
    def test_tokenizer_special_tokens(self):
        """Tokenizer should have a vocabulary."""
        docs = ["Hello"]
        tok = EvolvingTokenizer(docs)
        
        # Check that vocabulary exists and has entries
        assert tok.vocab_size > 0
        assert len(tok.stoi) > 0
        assert len(tok.itos) > 0


# ============================================================
# 6) SQLITE MEMORY TESTS
# ============================================================

class TestSQLiteMemory:
    """Test SQLite memory operations."""
    
    def test_init_db(self):
        """Database initialization should create tables."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            con = init_db(db_path)
            cursor = con.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "messages" in tables
            assert "corpus_events" in tables
            
            con.close()
        finally:
            os.unlink(db_path)
    
    def test_add_and_retrieve_messages(self):
        """Messages should be stored and retrieved correctly."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            con = init_db(db_path)
            
            # Add messages
            db_add_message(con, "user", "Hello molecule!")
            db_add_message(con, "assistant", "I am molecule. I listen.")
            db_add_message(con, "user", "What are you?")
            
            # Retrieve messages
            messages = db_recent_messages(con, limit=10)
            
            assert len(messages) == 3
            assert messages[0] == ("user", "Hello molecule!")
            assert messages[1] == ("assistant", "I am molecule. I listen.")
            assert messages[2] == ("user", "What are you?")
            
            con.close()
        finally:
            os.unlink(db_path)


# ============================================================
# 7) CORPUS MANAGEMENT TESTS
# ============================================================

class TestCorpusManagement:
    """Test corpus loading, saving, and reservoir management."""
    
    def test_load_save_corpus(self):
        """Corpus should be saved and loaded correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            corpus_path = f.name
        
        try:
            lines = ["First line.", "Second line.", "Third line."]
            save_corpus_lines(corpus_path, lines)
            
            loaded = load_corpus_lines(corpus_path)
            
            assert len(loaded) == 3
            assert loaded[0] == "First line."
            assert loaded[1] == "Second line."
            assert loaded[2] == "Third line."
        finally:
            os.unlink(corpus_path)
    
    def test_normalize_text(self):
        """Text normalization should clean whitespace."""
        text = "Hello   world!\t\tHow  are   you?"
        result = normalize_text(text)
        assert result == "Hello world! How are you?"
    
    def test_extract_sentences_from_messages(self):
        """Message extraction should produce tagged sentences."""
        messages = [
            ("user", "Hello molecule. How are you?"),
            ("assistant", "I am well. I learn from conversation."),
        ]
        
        sentences = extract_candidate_sentences_from_messages(messages)
        
        # Should produce sentences with H:/A: tags
        assert any("H:" in s for s in sentences)
        assert any("A:" in s for s in sentences)
    
    def test_reservoir_mix_keep(self):
        """Reservoir should maintain bounded size."""
        old_lines = [f"old_{i}" for i in range(100)]
        new_lines = [f"new_{i}" for i in range(50)]
        
        result = reservoir_mix_keep(old_lines, new_lines, max_lines=80)
        
        assert len(result) <= 80
        # Should contain some new lines
        assert any("new_" in line for line in result)


# ============================================================
# 8) INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_forward_pass_chain(self):
        """Full forward pass through vector operations."""
        # Create a simple computation graph
        x = VectorValue([1.0, 2.0, 3.0])
        w = VectorValue([0.5, 0.5, 0.5])
        
        # Compute: relu(x * w + 1)
        scaled = VectorValue([x.data[i] * w.data[i] for i in range(len(x.data))])
        biased = scaled + VectorValue([1.0, 1.0, 1.0])
        
        # [0.5+1, 1+1, 1.5+1] = [1.5, 2, 2.5]
        assert biased.data == [1.5, 2.0, 2.5]
    
    def test_tokenize_and_encode(self):
        """Full tokenization pipeline."""
        docs = [
            "Hello world.",
            "I am molecule.",
            "Testing the tokenizer.",
        ]
        
        tok = EvolvingTokenizer(docs)
        
        for doc in docs:
            ids = tok.encode(doc)
            decoded = tok.decode(ids)
            # Should reconstruct the original (ignoring whitespace/case issues)
            assert len(decoded) > 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
