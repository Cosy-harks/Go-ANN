// Package act defines activation functions and the derivatives of them
package act

import (
	"math"
	"strings"
)

// Activation functions implemented
// Linear   2
// Sigmoid  3
// ReLU     4
// SoftSign 1

//ActivationFactory takes the 'name' of the function to be used and returns the corresponding struct
//Linear
//sigmoid
//ReLU
//SoftSign
//default is ReLU
func ActivationFactory(method string) Activation {
	switch strings.ToLower(method) {
	case "linear":
		return Linear{}
	case "sigmoid":
		return Sigmoid{}
	case "relu":
		return ReLU{}
	case "softsign":
		return SoftSign{}
	default:
		return ReLU{}
	}
}

// Activation provides an interface to follow for making
// different versions of activation methods
type Activation interface {
	Act([]float64) []float64
	Derivative([]float64) []float64
	ToString() string
}

// SoftSign implements the activation interface
// using f(x) = x/(1 + |x|)
// It is tanh like (~-1.0?, ~+1.0?)
type SoftSign struct{}

func (s SoftSign) ToString() string {
	return "SoftSign"
}

// Act of SoftSign makes a smooth transition (-1, 1)
func (s SoftSign) Act(x []float64) []float64 {
	guess := make([]float64, len(x), len(x))
	for i, x := range x {
		guess[i] = x / (1.0 + math.Abs(x))
	}
	return guess
}

// Derivative of SoftSign makes a smooth transition (-1, 1)
func (s SoftSign) Derivative(y []float64) []float64 {
	v := make([]float64, len(y), len(y))
	for i, y := range y {
		v[i] = math.Pow(1.0/(1.0+math.Abs(y)), 2.0)
	}
	return v
}

// Linear implements the activation interface
// using a linear function
type Linear struct{}

func (l Linear) ToString() string {
	return "Linear"
}

// Act of Linear technically doesn't change the input data, but does return it.
func (l Linear) Act(x []float64) []float64 {
	guess := make([]float64, len(x), len(x))
	for i, v := range x {
		guess[i] = v
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

// Sigmoid implments the activation interface
// using a smooth transition (0, 1)
type Sigmoid struct{}

func (S Sigmoid) ToString() string {
	return "Sigmoid"
}

// Act on x suchthat y = 1/(1+e^-x))
func (S Sigmoid) Act(x []float64) []float64 {
	guess := make([]float64, len(x), len(x))
	for i, v := range x {
		guess[i] = (1. / (1. + math.Pow(math.E, -v)))
	}
	return guess
}

// Derivative on y suchthat y` = y(1. - y)
func (S Sigmoid) Derivative(y []float64) []float64 {
	v := make([]float64, len(y), len(y))
	for i, y := range y {
		v[i] = y * (1. - y)
	}
	return v
}

// ReLU implments the activation interface
// using a hard transition y = {0; x < 0, x; x >= 0}
type ReLU struct{}

func (r ReLU) ToString() string {
	return "ReLU"
}

// Act of Rectified Linear Unit gives max of (0, x)
func (r ReLU) Act(x []float64) []float64 {
	guess := make([]float64, len(x), len(x))
	for i, y := range x {
		if y >= 0 {
			guess[i] = y
		} else {
			guess[i] = 0.0
		}
	}
	return guess
}

// Derivative of Rectified Linear Unit give slope of (max(0, x)) == 0. || 1.
func (r ReLU) Derivative(y []float64) []float64 {
	v := make([]float64, len(y), len(y))
	for i, y := range y {
		if y >= 0 {
			v[i] = 1.0
		} else {
			v[i] = 0.0
		}
	}
	return v
}
