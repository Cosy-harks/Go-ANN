package Parts

import (
	"fmt"
	"math/rand"
)

//WeightedConnect provides access to consecutive layers
type WeightedConnect struct {
	Inlayer  *Layer // Getting an "undeclared name: Layer" compiler error - seems wrong
	Outlayer *Layer
	Weight   [][]float64
	Bias     []float64

	BatchWeightMod [][]float64
	BatchBiasMod   []float64
}

//Init
//inlayer | Weights | outlayer
//   [out][in] * [in] + [out] > I don't know what I was trying to describe
//First  indicies represent the outlayer
//Second indicies represent the inlayer
func (wc *WeightedConnect) Init() {
	wc.Weight = make([][]float64, wc.Outlayer.GetCount())
	wc.BatchWeightMod = make([][]float64, wc.Outlayer.GetCount())

	for i := range wc.Weight {
		wc.Weight[i] = make([]float64, wc.Inlayer.GetCount())
		wc.BatchWeightMod[i] = make([]float64, wc.Inlayer.GetCount())
	}

	wc.Bias = make([]float64, wc.Outlayer.GetCount())
	wc.BatchBiasMod = make([]float64, wc.Outlayer.GetCount())

	for i := range wc.Weight {
		for j := range wc.Weight[i] {
			wc.Weight[i][j] = rand.Float64()*2. - 1.
		}
		wc.Bias[i] = rand.Float64()
	}
}

// NodeAdded increases the Weight and bias slices to accomodate the node extention
func (wc *WeightedConnect) NodeAdded(outLayerIncreased bool, inLayerIncreased bool) {
	if outLayerIncreased {
		var countInlayer = wc.Inlayer.GetCount()
		var countOutlayer = wc.Outlayer.GetCount()

		wc.Weight = append(wc.Weight, make([]float64, countInlayer))

		wc.Bias = append(wc.Bias, (rand.Float64()*2. - 1.))

		for i := 0; i < countInlayer; i++ {
			wc.Weight[countOutlayer-1][i] = rand.Float64()*2. - 1.
		}
	}

	if inLayerIncreased {
		var countOutlayer = wc.Outlayer.GetCount()

		for i := 0; i < countOutlayer; i++ {
			wc.Weight[i] = append(wc.Weight[i], (rand.Float64()*2. - 1.))
		}
	}
}

//NodeRemoved decreases the weight and bias slice to accomodate the change of Network shape
func (wc *WeightedConnect) NodeRemoved(outLayerDecreased, inLayerDecreased bool, node int) {
	//Might be working
	if outLayerDecreased && node < len(wc.Weight) {
		var WeightSliceL = wc.Weight[:node]
		var WeightSliceR = wc.Weight[node+1:]

		wc.Weight = append(WeightSliceL, WeightSliceR...)
		wc.Bias = append(wc.Bias[:node], wc.Bias[node+1:]...)
	}

	if inLayerDecreased && node < len(wc.Weight[0]) {
		var countInlayer = wc.Inlayer.GetCount()
		var countOutlayer = wc.Outlayer.GetCount()

		for i := 0; i < countOutlayer; i++ {
			wc.Weight[i] = append(wc.Weight[i][:node], wc.Weight[i][node+1:countInlayer]...)
		}
	}
}

// Propagate moves the data along the network by one layer
func (wc *WeightedConnect) Propagate() {
	pre := wc.mathy()

	guess := wc.Outlayer.Style.Act(pre)

	err := wc.Outlayer.SetInout(guess)
	// I don't have experience with go errors
	if err != nil {
		fmt.Println(err.Error())
		panic(err.Error())
	}
}

//mathy does slow matrix multiplication
//Inlayer[] * weight[][] + bias[] -> return
func (wc *WeightedConnect) mathy() []float64 {
	guess := make([]float64, wc.Outlayer.GetCount())
	for i, v := range wc.Weight {
		for j, w := range v {
			guess[i] += wc.Inlayer.Inout[j] * w
		}
		guess[i] += wc.Bias[i]
	}
	return guess
}

//Brakagate is for backpropagation. It takes as input the 'outside derivative' of the specific layer
// and returns the outside derivative for the layer preceeding it.
//	Ex:
/*
// terr[] gets 2 * error[] * activation.Derivative(guess[])[]
for i := len(n.Connectors) - 1; i >= 0; i-- {
	terr = n.Connectors[i].brakagate(terr)
}
*/
func (wc *WeightedConnect) Brakagate(outDer []float64) []float64 {
	//Setup slices that match the weight and bias shapes
	var weightMod [][]float64
	var biasMod []float64
	var inDer []float64
	weightMod = make([][]float64, len(wc.Weight))
	biasMod = make([]float64, len(outDer))
	inDer = make([]float64, wc.Inlayer.GetCount())

	//Derivative for this layer
	weightMod = wc.midDirivative(outDer)
	biasMod = wc.biasDirivative(outDer)
	partDer := wc.inDerivative(weightMod)

	//Weight values modified
	wc.addBatchModValues(weightMod, biasMod)

	wholeDer := make([]float64, 0)
	inDer = wc.Inlayer.Style.Derivative(wc.Inlayer.GetInout())

	for j := range inDer {
		wholeDer = append(wholeDer, (inDer[j] * partDer[j]))
	}

	//Return the outDerivative for the preceeding layer
	return wholeDer
}

//midDerivative multiplies the derivative and input layer
// return [][] of weight modifications
func (wc *WeightedConnect) midDirivative(outDer []float64) [][]float64 {
	weightMod := make([][]float64, len(wc.Weight))

	for i := range weightMod {
		weightMod[i] = make([]float64, len(wc.Weight[i]))
	}

	for i, derv := range outDer {
		for j, ins := range wc.Inlayer.Inout {
			weightMod[i][j] = derv * ins
		}
	}
	return weightMod
}

//biasDerivative return duplicate of the outside derivative.
func (wc *WeightedConnect) biasDirivative(outDer []float64) []float64 {
	biasMod := make([]float64, len(outDer))

	for i, derv := range outDer {
		biasMod[i] = derv * 1.0
	}

	return biasMod
}

// inDerivative will create a slice to pass to the activation.derivative() to the next brakagate
func (wc *WeightedConnect) inDerivative(weightMod [][]float64) []float64 {

	inDer := make([]float64, wc.Inlayer.GetCount())
	for i := range wc.Weight {
		for j := 0; j < len(inDer); j++ {
			inDer[j] += wc.Weight[i][j] * weightMod[i][j]
		}
	}

	return inDer
}

//addModeValues
func (wc *WeightedConnect) addBatchModValues(weightMod [][]float64, biasMod []float64) {
	//fmt.Println()
	for i := range wc.Weight {
		for j := range wc.Weight[i] {
			wc.BatchWeightMod[i][j] += weightMod[i][j]
		}
	}
	//fmt.Println("wm: ", weightMod)
	//fmt.Println()

	for i := range wc.Bias {
		wc.BatchBiasMod[i] += biasMod[i]
	}
}

//DistributeBatchMod updates Weight values with relative adjustment
// and clears the Batch variables to 0.
func (wc *WeightedConnect) DistributeBatchMod(learnRate float64) {
	for i := range wc.Weight {
		for j := range wc.Weight[i] {
			wc.Weight[i][j] -= wc.BatchWeightMod[i][j] * learnRate
			wc.BatchWeightMod[i][j] = 0.
		}
	}

	for i := range wc.Bias {
		wc.Bias[i] -= wc.BatchBiasMod[i] * learnRate
		wc.BatchBiasMod[i] = 0.
	}
}
