package main

import (
	"fmt"

	"github.com/Go-ANN/act"
)

func main() {
	fmt.Println(act.ActivationFactory("linear ").Act([]float64{1.1, 2.3, 0.89, -2.}))
}
