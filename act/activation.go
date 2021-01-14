// Package act defines activation functions and the derivatives of them
package act

import "math"

// Activation provides an interface to follow for making
// different versions of activation methods
type Activation interface {
	Act([]float64) []float64
	Derivative([]float64) []float64
}

// Linear implments the activation interface
// using a linear function
type Linear struct{}

// Sigmoid implments the activation interface
// using a smooth transition (0, 1)
type Sigmoid struct{}

// Act of Linear technically doesn't change the input data, but does return it.
func (l Linear) Act(x []float64) []float64 {
	guess := make([]float64, len(x), len(x))
	for i, v := range x {
		guess[i] = v
	}
	return guess
}

// Act on x suchthat y = 1/(1+e^-x))
func (S Sigmoid) Act(x []float64) []float64 {
	guess := make([]float64, len(x), len(x))
	for i, v := range x {
		guess[i] = (1. / (1. + math.Pow(math.E, -v)))
	}
	return guess
}

// Derivative of Linear returns the scalar of the linear function. 1.0
func (l Linear) Derivative(x []float64) []float64 {
	v := make([]float64, len(x), len(x))
	for i := range x {
		v[i] = 1.0
	}
	return v
}

// Derivative on y suchthat y` = y(1. - y)
func (S Sigmoid) Derivative(y []float64) []float64 {
	v := make([]float64, len(y), len(y))
	for i, y := range y {
		v[i] = y * (1. - y)
	}
	return v
}
