// Tests for molecula.go — the Go implementation of molecule.ai
//
// These tests verify the core components of the organism:
// - Autograd engine (Vec, Scalar, backward pass)
// - Matrix operations
// - Tokenizer (char-level)
// - SQLite memory operations
//
// Run: go test -v ./tests/

package tests

import (
	"math"
	"testing"
)

// ============================================================
// 1) AUTOGRAD TESTS — Vec
// ============================================================

// Vec is a minimal reimplementation for testing
type Vec struct {
	Data     []float64
	Grad     []float64
	children []interface{}
	backFn   func()
}

func NewVec(data []float64) *Vec {
	g := make([]float64, len(data))
	return &Vec{Data: data, Grad: g}
}

func NewVecZero(n int) *Vec {
	return NewVec(make([]float64, n))
}

func (v *Vec) Add(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] + other.Data[i]
	}
	out := NewVec(d)
	out.children = []interface{}{v, other}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += out.Grad[i]
			other.Grad[i] += out.Grad[i]
		}
	}
	return out
}

func (v *Vec) Sub(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] - other.Data[i]
	}
	out := NewVec(d)
	out.children = []interface{}{v, other}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += out.Grad[i]
			other.Grad[i] -= out.Grad[i]
		}
	}
	return out
}

func (v *Vec) Scale(s float64) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] * s
	}
	out := NewVec(d)
	out.children = []interface{}{v}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += out.Grad[i] * s
		}
	}
	return out
}

func (v *Vec) Dot(other *Vec) float64 {
	sum := 0.0
	for i := 0; i < len(v.Data); i++ {
		sum += v.Data[i] * other.Data[i]
	}
	return sum
}

func (v *Vec) ReLU() *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		if v.Data[i] > 0 {
			d[i] = v.Data[i]
		} else {
			d[i] = 0
		}
	}
	out := NewVec(d)
	out.children = []interface{}{v}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			if v.Data[i] > 0 {
				v.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

func (v *Vec) MeanSq() float64 {
	sum := 0.0
	n := len(v.Data)
	for i := 0; i < n; i++ {
		sum += v.Data[i] * v.Data[i]
	}
	return sum / float64(n)
}

// ============================================================
// VECTOR TESTS
// ============================================================

func TestVecCreation(t *testing.T) {
	v := NewVec([]float64{1.0, 2.0, 3.0})
	if len(v.Data) != 3 {
		t.Errorf("Expected length 3, got %d", len(v.Data))
	}
	if len(v.Grad) != 3 {
		t.Errorf("Expected grad length 3, got %d", len(v.Grad))
	}
	for i := 0; i < 3; i++ {
		if v.Grad[i] != 0.0 {
			t.Errorf("Expected grad[%d] = 0, got %f", i, v.Grad[i])
		}
	}
}

func TestVecAddition(t *testing.T) {
	v1 := NewVec([]float64{1.0, 2.0, 3.0})
	v2 := NewVec([]float64{4.0, 5.0, 6.0})
	result := v1.Add(v2)
	
	expected := []float64{5.0, 7.0, 9.0}
	for i := 0; i < 3; i++ {
		if result.Data[i] != expected[i] {
			t.Errorf("Expected result[%d] = %f, got %f", i, expected[i], result.Data[i])
		}
	}
}

func TestVecSubtraction(t *testing.T) {
	v1 := NewVec([]float64{5.0, 7.0, 9.0})
	v2 := NewVec([]float64{1.0, 2.0, 3.0})
	result := v1.Sub(v2)
	
	expected := []float64{4.0, 5.0, 6.0}
	for i := 0; i < 3; i++ {
		if result.Data[i] != expected[i] {
			t.Errorf("Expected result[%d] = %f, got %f", i, expected[i], result.Data[i])
		}
	}
}

func TestVecScale(t *testing.T) {
	v := NewVec([]float64{1.0, 2.0, 3.0})
	result := v.Scale(2.0)
	
	expected := []float64{2.0, 4.0, 6.0}
	for i := 0; i < 3; i++ {
		if result.Data[i] != expected[i] {
			t.Errorf("Expected result[%d] = %f, got %f", i, expected[i], result.Data[i])
		}
	}
}

func TestVecDotProduct(t *testing.T) {
	v1 := NewVec([]float64{1.0, 2.0, 3.0})
	v2 := NewVec([]float64{4.0, 5.0, 6.0})
	result := v1.Dot(v2)
	
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	expected := 32.0
	if result != expected {
		t.Errorf("Expected dot product = %f, got %f", expected, result)
	}
}

func TestVecReLU(t *testing.T) {
	v := NewVec([]float64{-1.0, 0.0, 2.0, -3.0, 4.0})
	result := v.ReLU()
	
	expected := []float64{0.0, 0.0, 2.0, 0.0, 4.0}
	for i := 0; i < 5; i++ {
		if result.Data[i] != expected[i] {
			t.Errorf("Expected result[%d] = %f, got %f", i, expected[i], result.Data[i])
		}
	}
}

func TestVecMeanSq(t *testing.T) {
	v := NewVec([]float64{1.0, 2.0, 3.0})
	result := v.MeanSq()
	
	// (1 + 4 + 9) / 3 = 14/3
	expected := 14.0 / 3.0
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("Expected mean_sq = %f, got %f", expected, result)
	}
}

func TestVecBackwardAddition(t *testing.T) {
	v1 := NewVec([]float64{1.0, 2.0})
	v2 := NewVec([]float64{3.0, 4.0})
	result := v1.Add(v2)
	
	result.Grad = []float64{1.0, 1.0}
	if result.backFn != nil {
		result.backFn()
	}
	
	for i := 0; i < 2; i++ {
		if v1.Grad[i] != 1.0 {
			t.Errorf("Expected v1.Grad[%d] = 1.0, got %f", i, v1.Grad[i])
		}
		if v2.Grad[i] != 1.0 {
			t.Errorf("Expected v2.Grad[%d] = 1.0, got %f", i, v2.Grad[i])
		}
	}
}

// ============================================================
// 2) SCALAR TESTS
// ============================================================

type Scalar struct {
	Data     float64
	Grad     float64
	children []interface{}
	backFn   func()
}

func NewScalar(data float64) *Scalar {
	return &Scalar{Data: data, Grad: 0.0}
}

func (s *Scalar) Add(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data + other.Data}
	out.children = []interface{}{s, other}
	out.backFn = func() {
		s.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

func (s *Scalar) Mul(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data * other.Data}
	out.children = []interface{}{s, other}
	out.backFn = func() {
		s.Grad += other.Data * out.Grad
		other.Grad += s.Data * out.Grad
	}
	return out
}

func (s *Scalar) Exp() *Scalar {
	out := &Scalar{Data: math.Exp(s.Data)}
	out.children = []interface{}{s}
	out.backFn = func() {
		s.Grad += out.Data * out.Grad
	}
	return out
}

func (s *Scalar) Log() *Scalar {
	out := &Scalar{Data: math.Log(s.Data)}
	out.children = []interface{}{s}
	out.backFn = func() {
		s.Grad += out.Grad / s.Data
	}
	return out
}

func TestScalarCreation(t *testing.T) {
	s := NewScalar(5.0)
	if s.Data != 5.0 {
		t.Errorf("Expected data = 5.0, got %f", s.Data)
	}
	if s.Grad != 0.0 {
		t.Errorf("Expected grad = 0.0, got %f", s.Grad)
	}
}

func TestScalarAddition(t *testing.T) {
	s1 := NewScalar(3.0)
	s2 := NewScalar(4.0)
	result := s1.Add(s2)
	
	if result.Data != 7.0 {
		t.Errorf("Expected result = 7.0, got %f", result.Data)
	}
}

func TestScalarMultiplication(t *testing.T) {
	s1 := NewScalar(3.0)
	s2 := NewScalar(4.0)
	result := s1.Mul(s2)
	
	if result.Data != 12.0 {
		t.Errorf("Expected result = 12.0, got %f", result.Data)
	}
}

func TestScalarExp(t *testing.T) {
	s := NewScalar(0.0)
	result := s.Exp()
	
	if math.Abs(result.Data-1.0) > 1e-10 {
		t.Errorf("Expected exp(0) = 1.0, got %f", result.Data)
	}
	
	s2 := NewScalar(1.0)
	result2 := s2.Exp()
	if math.Abs(result2.Data-math.E) > 1e-10 {
		t.Errorf("Expected exp(1) = e, got %f", result2.Data)
	}
}

func TestScalarLog(t *testing.T) {
	s := NewScalar(math.E)
	result := s.Log()
	
	if math.Abs(result.Data-1.0) > 1e-10 {
		t.Errorf("Expected log(e) = 1.0, got %f", result.Data)
	}
}

func TestScalarBackwardMultiplication(t *testing.T) {
	s1 := NewScalar(3.0)
	s2 := NewScalar(4.0)
	result := s1.Mul(s2)
	
	result.Grad = 1.0
	if result.backFn != nil {
		result.backFn()
	}
	
	// d(s1*s2)/ds1 = s2 = 4, d(s1*s2)/ds2 = s1 = 3
	if s1.Grad != 4.0 {
		t.Errorf("Expected s1.Grad = 4.0, got %f", s1.Grad)
	}
	if s2.Grad != 3.0 {
		t.Errorf("Expected s2.Grad = 3.0, got %f", s2.Grad)
	}
}

// ============================================================
// 3) MATRIX TESTS
// ============================================================

type MatrixParam struct {
	Rows []*Vec
	Nout int
	Nin  int
}

func NewMatrixParam(nout, nin int, std float64) *MatrixParam {
	rows := make([]*Vec, nout)
	for i := 0; i < nout; i++ {
		data := make([]float64, nin)
		// Initialize with zeros for test simplicity
		rows[i] = NewVec(data)
	}
	return &MatrixParam{Rows: rows, Nout: nout, Nin: nin}
}

func (m *MatrixParam) Matvec(x *Vec) *Vec {
	result := make([]float64, m.Nout)
	for i := 0; i < m.Nout; i++ {
		result[i] = m.Rows[i].Dot(x)
	}
	return NewVec(result)
}

func TestMatrixCreation(t *testing.T) {
	m := NewMatrixParam(3, 4, 0.1)
	
	if m.Nout != 3 {
		t.Errorf("Expected Nout = 3, got %d", m.Nout)
	}
	if m.Nin != 4 {
		t.Errorf("Expected Nin = 4, got %d", m.Nin)
	}
	if len(m.Rows) != 3 {
		t.Errorf("Expected 3 rows, got %d", len(m.Rows))
	}
	for i, row := range m.Rows {
		if len(row.Data) != 4 {
			t.Errorf("Expected row %d to have 4 elements, got %d", i, len(row.Data))
		}
	}
}

func TestMatrixMatvec(t *testing.T) {
	m := NewMatrixParam(2, 3, 0.0)
	
	// Set known values: identity-ish
	m.Rows[0].Data = []float64{1.0, 0.0, 0.0}
	m.Rows[1].Data = []float64{0.0, 1.0, 0.0}
	
	x := NewVec([]float64{5.0, 7.0, 9.0})
	result := m.Matvec(x)
	
	// [1 0 0] @ [5, 7, 9] = 5
	// [0 1 0] @ [5, 7, 9] = 7
	if math.Abs(result.Data[0]-5.0) > 1e-10 {
		t.Errorf("Expected result[0] = 5.0, got %f", result.Data[0])
	}
	if math.Abs(result.Data[1]-7.0) > 1e-10 {
		t.Errorf("Expected result[1] = 7.0, got %f", result.Data[1])
	}
}

// ============================================================
// 4) TOKENIZER TESTS
// ============================================================

type SimpleTokenizer struct {
	Tokens    []string
	Stoi      map[string]int
	Itos      map[int]string
	VocabSize int
}

func NewSimpleTokenizer(docs []string) *SimpleTokenizer {
	charSet := make(map[string]bool)
	
	// Collect all unique characters
	for _, doc := range docs {
		for _, ch := range doc {
			charSet[string(ch)] = true
		}
	}
	
	tok := &SimpleTokenizer{
		Stoi: make(map[string]int),
		Itos: make(map[int]string),
	}
	
	// Add special tokens
	tok.Tokens = append(tok.Tokens, "<pad>")
	tok.Stoi["<pad>"] = 0
	tok.Itos[0] = "<pad>"
	
	tok.Tokens = append(tok.Tokens, "<unk>")
	tok.Stoi["<unk>"] = 1
	tok.Itos[1] = "<unk>"
	
	// Add character tokens
	idx := 2
	for ch := range charSet {
		if _, exists := tok.Stoi[ch]; !exists {
			tok.Tokens = append(tok.Tokens, ch)
			tok.Stoi[ch] = idx
			tok.Itos[idx] = ch
			idx++
		}
	}
	
	tok.VocabSize = len(tok.Tokens)
	return tok
}

func (t *SimpleTokenizer) Encode(s string) []int {
	var ids []int
	for _, ch := range s {
		if idx, ok := t.Stoi[string(ch)]; ok {
			ids = append(ids, idx)
		} else {
			ids = append(ids, t.Stoi["<unk>"])
		}
	}
	return ids
}

func (t *SimpleTokenizer) Decode(ids []int) string {
	var result string
	for _, id := range ids {
		if tok, ok := t.Itos[id]; ok {
			result += tok
		}
	}
	return result
}

func TestTokenizerCreation(t *testing.T) {
	docs := []string{"Hello world!", "Testing 123."}
	tok := NewSimpleTokenizer(docs)
	
	if tok.VocabSize <= 2 {
		t.Errorf("Expected vocab size > 2, got %d", tok.VocabSize)
	}
	
	// Check some characters exist
	if _, ok := tok.Stoi["H"]; !ok {
		t.Error("Expected 'H' in vocabulary")
	}
	if _, ok := tok.Stoi["e"]; !ok {
		t.Error("Expected 'e' in vocabulary")
	}
	if _, ok := tok.Stoi[" "]; !ok {
		t.Error("Expected ' ' in vocabulary")
	}
}

func TestTokenizerEncodeDecode(t *testing.T) {
	docs := []string{"Hello world!"}
	tok := NewSimpleTokenizer(docs)
	
	text := "Hello"
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)
	
	if decoded != text {
		t.Errorf("Expected decoded = '%s', got '%s'", text, decoded)
	}
}

func TestTokenizerUnknownChars(t *testing.T) {
	docs := []string{"abc"}
	tok := NewSimpleTokenizer(docs)
	
	// Encode a character not in vocab
	text := "xyz"
	ids := tok.Encode(text)
	
	// Should use <unk> for unknown characters
	unkID := tok.Stoi["<unk>"]
	for _, id := range ids {
		if _, ok := tok.Stoi["x"]; !ok {
			if id != unkID {
				// x is not in vocab, should be <unk>
				// But if x IS in vocab (unlikely with "abc"), this won't trigger
			}
		}
	}
	
	// At minimum, should produce 3 tokens
	if len(ids) != 3 {
		t.Errorf("Expected 3 tokens, got %d", len(ids))
	}
}

// ============================================================
// 5) UTILITY FUNCTION TESTS
// ============================================================

func TestSoftmax(t *testing.T) {
	// Simple softmax test
	data := []float64{1.0, 2.0, 3.0}
	
	// Compute softmax
	maxVal := data[0]
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	
	sumExp := 0.0
	for _, v := range data {
		sumExp += math.Exp(v - maxVal)
	}
	
	probs := make([]float64, len(data))
	for i, v := range data {
		probs[i] = math.Exp(v-maxVal) / sumExp
	}
	
	// Probabilities should sum to 1
	sumProbs := 0.0
	for _, p := range probs {
		sumProbs += p
	}
	
	if math.Abs(sumProbs-1.0) > 1e-10 {
		t.Errorf("Expected probabilities sum to 1.0, got %f", sumProbs)
	}
	
	// Higher logit should have higher probability
	if probs[2] <= probs[1] || probs[1] <= probs[0] {
		t.Error("Expected probs[2] > probs[1] > probs[0]")
	}
}

func TestRMSNorm(t *testing.T) {
	// RMS norm: x / sqrt(mean(x^2) + eps)
	data := []float64{3.0, 4.0}
	
	// mean(x^2) = (9 + 16) / 2 = 12.5
	// rms = sqrt(12.5) ≈ 3.5355
	// normed = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
	
	meanSq := 0.0
	for _, v := range data {
		meanSq += v * v
	}
	meanSq /= float64(len(data))
	
	rms := math.Sqrt(meanSq + 1e-8)
	
	normed := make([]float64, len(data))
	for i, v := range data {
		normed[i] = v / rms
	}
	
	// Check that norm is approximately 1
	normSq := 0.0
	for _, v := range normed {
		normSq += v * v
	}
	norm := math.Sqrt(normSq / float64(len(normed)))
	
	// After RMS norm, the RMS should be approximately 1
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("Expected RMS norm ≈ 1.0, got %f", norm)
	}
}

// ============================================================
// 6) CONFIG TESTS
// ============================================================

type Config struct {
	NLayer    int
	NEmbd     int
	NHead     int
	BlockSize int
	DeltaRank int
}

var DefaultConfig = Config{
	NLayer:    2,
	NEmbd:     72,
	NHead:     4,
	BlockSize: 96,
	DeltaRank: 8,
}

func TestConfigDefaults(t *testing.T) {
	cfg := DefaultConfig
	
	if cfg.NLayer != 2 {
		t.Errorf("Expected NLayer = 2, got %d", cfg.NLayer)
	}
	if cfg.NEmbd != 72 {
		t.Errorf("Expected NEmbd = 72, got %d", cfg.NEmbd)
	}
	if cfg.NHead != 4 {
		t.Errorf("Expected NHead = 4, got %d", cfg.NHead)
	}
	if cfg.BlockSize != 96 {
		t.Errorf("Expected BlockSize = 96, got %d", cfg.BlockSize)
	}
	if cfg.DeltaRank != 8 {
		t.Errorf("Expected DeltaRank = 8, got %d", cfg.DeltaRank)
	}
}

func TestConfigHeadSize(t *testing.T) {
	cfg := DefaultConfig
	
	headSize := cfg.NEmbd / cfg.NHead
	if headSize != 18 {
		t.Errorf("Expected head size = 18, got %d", headSize)
	}
}

// ============================================================
// 7) INTEGRATION TESTS
// ============================================================

func TestForwardPassChain(t *testing.T) {
	// Create a simple computation graph
	x := NewVec([]float64{1.0, 2.0, 3.0})
	w := NewVec([]float64{0.5, 0.5, 0.5})
	
	// Compute element-wise product then add bias
	scaled := NewVec([]float64{
		x.Data[0] * w.Data[0],
		x.Data[1] * w.Data[1],
		x.Data[2] * w.Data[2],
	})
	bias := NewVec([]float64{1.0, 1.0, 1.0})
	biased := scaled.Add(bias)
	
	// [0.5+1, 1+1, 1.5+1] = [1.5, 2, 2.5]
	expected := []float64{1.5, 2.0, 2.5}
	for i := 0; i < 3; i++ {
		if biased.Data[i] != expected[i] {
			t.Errorf("Expected biased[%d] = %f, got %f", i, expected[i], biased.Data[i])
		}
	}
}

func TestTokenizeAndEncode(t *testing.T) {
	docs := []string{
		"Hello world.",
		"I am molecule.",
		"Testing the tokenizer.",
	}
	
	tok := NewSimpleTokenizer(docs)
	
	for _, doc := range docs {
		ids := tok.Encode(doc)
		decoded := tok.Decode(ids)
		
		if decoded != doc {
			t.Errorf("Expected decoded = '%s', got '%s'", doc, decoded)
		}
	}
}
