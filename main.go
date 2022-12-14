package main

import (
	"fmt"
)

type LinearRegression struct {
	n      int
	p      int
	Y      [][]float64
	Y_pred [][]float64
	Coeff  [][]float64
	X      [][]float64
	SSE    float64
	SSR    float64
	SST    float64
}

func (LinearRegression LinearRegression) fit(matrix1 [][]float64, matrix2 [][]float64) LinearRegression {
	LinearRegression.X = matrix1
	LinearRegression.Y = matrix2
	LinearRegression.n = len(matrix1)
	LinearRegression.p = len(matrix1[0])
	LinearRegression.Coeff = beta(matrix1, matrix2)
	LinearRegression.Y_pred = MulMatrix(Augment(oneMatrix(len(matrix1)), matrix1), LinearRegression.Coeff)
	LinearRegression.SSE = SC(mean(LinearRegression.Y), LinearRegression.Y_pred)
	LinearRegression.SSR = SC(LinearRegression.Y, LinearRegression.Y_pred)
	LinearRegression.SST = LinearRegression.SSR + LinearRegression.SSE
	return LinearRegression
}
func (LinearRegression LinearRegression) printResult() {
	fmt.Println("intercept :	", LinearRegression.Coeff[0][0])
	for i := 1; i < len(LinearRegression.Coeff); i++ {
		fmt.Println("coefficient ", i, " :	", LinearRegression.Coeff[i][0])
	}
	fmt.Println("SSE :	", LinearRegression.SSE)
	fmt.Println("SSR :	", LinearRegression.SSR)
	fmt.Println("SST :	", LinearRegression.SST)
}

func (LinearRegression LinearRegression) getCoefficient() [][]float64 {
	return LinearRegression.Coeff
}
func (LinearRegression LinearRegression) y_pred() []float64 {
	return LinearRegression.y_pred()
}

func SC(matrix1 [][]float64, matrix2 [][]float64) float64 {
	var W = [][]float64{SubMatrix(trp(matrix1)[0], trp(matrix2)[0])}
	return MulMatrix(W, trp(W))[0][0]
}
func mean(matrix [][]float64) [][]float64 {
	X := trp(matrix)[0]
	s := 0.
	for i := 0; i < len(X); i++ {
		s = s + X[i]
	}
	s = s / float64(len(X))
	return trp([][]float64{ScalMatrix(trp(oneMatrix(len(X)))[0], s)})
}
func AddMatrix(matrix1 [][]float64, matrix2 [][]float64) [][]float64 {
	result := make([][]float64, len(matrix1))
	for i, a := range matrix1 {
		for j := range a {
			result[i] = append(result[i], matrix1[i][j]+matrix2[i][j])
		}
	}
	return result
}

func SubMatrix(matrix1 []float64, matrix2 []float64) []float64 {
	result := make([]float64, len(matrix1))
	for i := 0; i < len(matrix1); i++ {
		result[i] = matrix1[i] - matrix2[i]
	}
	return result
}
func ScalMatrix(matrix1 []float64, r float64) []float64 {
	result := make([]float64, len(matrix1))
	for i := range matrix1 {
		result[i] = matrix1[i] * r
	}
	return result
}

func MulMatrix(matrix1 [][]float64, matrix2 [][]float64) [][]float64 {
	result := make([][]float64, len(matrix1))
	for i := range matrix1 {
		result[i] = make([]float64, len(matrix2[0]))
		for j := range matrix2[0] {
			for k := range matrix2 {
				result[i][j] += matrix1[i][k] * matrix2[k][j]
			}
		}
	}
	return result
}
func identityMatrix(size int) [][]float64 {
	m := make([][]float64, size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			m[i] = append(m[i], 0)
		}
		m[i][i] = 1
	}
	return m
}
func oneMatrix(size int) [][]float64 {
	m := make([][]float64, size)
	for i := 0; i < size; i++ {
		m[i] = append(m[i], 1)
	}
	return m
}

func Augment(m [][]float64, right [][]float64) [][]float64 {

	result := make([][]float64, len(m))
	for i := 0; i < len(m); i++ {
		result[i] = make([]float64, len(m[0])+len(right[0]))
	}
	for r, row := range m {
		for c := range row {
			result[r][c] = m[r][c]
		}
		cols := len(m[0])
		for c := range right[0] {
			result[r][cols+c] = right[r][c]
		}
	}
	return result
}

func Invert(m [][]float64) [][]float64 {
	n := len(m)
	M := Augment(m, identityMatrix(n))
	for i := 0; i < n; i++ {
		pivot := M[i][i]
		row := M[i]
		M[i] = ScalMatrix(row, 1/pivot)
		for k := 0; k < len(m); k++ {
			if k != i {
				M[k] = SubMatrix(M[k], ScalMatrix(M[i], M[k][i]))
			}
		}
	}
	result := make([][]float64, n)
	for i := 0; i < len(m); i++ {
		result[i] = make([]float64, n)
	}
	for j := 0; j < n; j++ {
		for k := 0; k < n; k++ {
			result[j][k] = M[j][k+n]
		}
	}
	return result
}
func trp(slice [][]float64) [][]float64 {
	xl := len(slice[0])
	yl := len(slice)
	result := make([][]float64, xl)
	for i := range result {
		result[i] = make([]float64, yl)
	}
	for i := 0; i < xl; i++ {
		for j := 0; j < yl; j++ {
			result[i][j] = slice[j][i]
		}
	}
	return result
}
func beta(X [][]float64, Y [][]float64) [][]float64 {
	X1 := Augment(oneMatrix(len(X)), X)
	return MulMatrix(Invert(MulMatrix(trp(X1), X1)), MulMatrix(trp(X1), Y))
}

func main() {
	var X_data = [][]float64{{-5.1, -4, -3, 2.1, -1, 0, 0.9, 2, 3.02, 4, 5, 6, 7.09, 8, 8.98, 10, 11, 12, 1, 3, 14, 15, -11, -12, 29, 30, 1, 50, 455, 500}}
	var y_data = []float64{-9, -7.1, -5, 5, -1, 1, 3.1, 5, 7, 9, 11, 13, 15, 17, 18.95, 21, 23, 25, 3, 7, 29, 31, -21, -23, 59.1, 60.95, 3, 101, 909.9, 999.78}
	var model LinearRegression
	model = model.fit(trp(X_data), trp([][]float64{y_data}))
	model.printResult()
}
