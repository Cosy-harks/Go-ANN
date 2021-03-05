// Package goann implments a neural network builder and trainer and two Activations
package goann

import (
	"fmt"
	"math/rand"

	"github.com/Go-ANN/act"
)

type layer struct {
	style act.Activation
	inout []float64
}

func (l *layer) addNode() {
	l.inout = append(l.inout, 0.0)
}

type weightedConnect struct {
	inlayer  *layer
	outlayer *layer
	weight   [][]float64
	bias     []float64

	batchWeightMod [][]float64
	batchBiasMod   []float64
}

// init
//inlayer | weights | outlayer
// [out][in] * [in] + [out]
func (wc *weightedConnect) init() {
	wc.weight = make([][]float64, len(wc.outlayer.inout))
	wc.batchWeightMod = make([][]float64, len(wc.outlayer.inout))

	for i := range wc.weight {
		wc.weight[i] = make([]float64, len(wc.inlayer.inout))
		wc.batchWeightMod[i] = make([]float64, len(wc.inlayer.inout))
	}

	wc.bias = make([]float64, len(wc.outlayer.inout))
	wc.batchBiasMod = make([]float64, len(wc.outlayer.inout))

	for i := range wc.weight {
		for j := range wc.weight[i] {
			wc.weight[i][j] = rand.Float64()*2. - 1.
		}
		wc.bias[i] = rand.Float64()
	}
}

// node added, need to update weights
func (wc *weightedConnect) nodeAdded(outLayerIncreased bool, inLayerIncreased bool) {
	if outLayerIncreased {
		var countInlayer = len(wc.inlayer.inout)
		var countOutlayer = len(wc.outlayer.inout)

		wc.weight = append(wc.weight, make([]float64, countInlayer))
		wc.batchWeightMod = append(wc.batchWeightMod, make([]float64, countInlayer))

		wc.bias = append(wc.bias, (rand.Float64()*2. - 1.))
		wc.batchBiasMod = append(wc.batchBiasMod, 0.0)

		for i := 0; i < countInlayer; i++ {
			wc.weight[countOutlayer-1][i] = rand.Float64()*2. - 1.
		}
	}

	if inLayerIncreased {
		var countOutlayer = len(wc.outlayer.inout)

		for i := 0; i < countOutlayer; i++ {
			wc.weight[i] = append(wc.weight[i], (rand.Float64()*2. - 1.))
			wc.batchWeightMod[i] = append(wc.batchWeightMod[i], 0.0)
		}
	}
}

// propagate is not assigning correctly to variable for the first three iterations
// Many subsequent iterations do assign properly
func (wc *weightedConnect) propagate() {
	// Matrix multiplication weight [][] * out [] + bias[]
	//assign to next layer
	/*
	   _			_       _	_ _									_
	   |a00 a01 a02| _	  _ |c00| |a00*b00 + a01*b10 + a02*b20 + c00|
	   |a10 a11 a12| |b00| |c10| |a10*b00 + a11*b10 + a12*b20 + c10|
	   |a20 a21 a22|*|b10|+|c20|=|a20*b00 + a21*b10 + a22*b20 + c20|
	   |a30 a31 a32| |b20| |c30| |a30*b00 + a31*b10 + a32*b20 + c30|
	   _			_ _	  _	_	_ _									_

	*/
	//fmt.Println("input Pre-act", wc.inlayer.inout)
	guess := wc.mathy()
	//fmt.Println("output Pre-act ", wc.outlayer)

	// sometimes assigns sometimes does not assign
	wc.outlayer.inout = wc.outlayer.style.Act(guess)
	//fmt.Println("output Post-act ", wc.outlayer.inout)
	//fmt.Println()
}

func (wc *weightedConnect) mathy() []float64 {
	guess := make([]float64, len(wc.outlayer.inout), len(wc.outlayer.inout))
	for i, v := range wc.weight {
		for j, w := range v {
			// seems to be assigning properly
			guess[i] += w * wc.inlayer.inout[j]
		}
		guess[i] += wc.bias[i]
	}
	return guess
}

func (wc *weightedConnect) brakagate(outDer []float64) []float64 {

	//fmt.Println("1")
	weightMod := make([][]float64, len(wc.weight))
	biasMod := make([]float64, len(outDer))

	weightMod = wc.midDirivative(outDer)
	biasMod = wc.biasDirivative(outDer)

	//inDerivative
	//fmt.Println("3")
	inDer := make([]float64, len(wc.inlayer.inout), len(wc.inlayer.inout))
	//fmt.Printf("weight size: %vx%v | ", len(wc.weight),len(wc.weight[0]))
	//fmt.Println("l(wc.in.io): ", len(wc.inlayer.inout))

	inDer = wc.inDerivative(weightMod)

	//Weight modifier
	wc.addModValues(weightMod, biasMod)

	//fmt.Println("6")
	return inDer
}

func (wc *weightedConnect) midDirivative(outDer []float64) [][]float64 {
	weightMod := make([][]float64, len(wc.weight))

	for i := range weightMod {
		weightMod[i] = make([]float64, len(wc.weight[i]))
	}

	for i, derv := range outDer {
		for j, ins := range wc.inlayer.inout {
			weightMod[i][j] = derv * ins
			//fmt.Println("derv: ", derv,", ins: ", ins, ", t",i,j,": ", weightMod[i][j])
		}
	}
	return weightMod
}

func (wc *weightedConnect) biasDirivative(outDer []float64) []float64 {
	biasMod := make([]float64, len(outDer))

	for i, derv := range outDer {
		biasMod[i] = derv * 1.0
	}

	return biasMod
}

// inDerivative will create a slice to pass to the next brakagate
func (wc *weightedConnect) inDerivative(weightMod [][]float64) []float64 {
	inDer := make([]float64, len(wc.inlayer.inout), len(wc.inlayer.inout))
	//fmt.Printf("weight size: %vx%v | ", len(wc.weight),len(wc.weight[0]))
	//fmt.Println("l(wc.in.io): ", len(wc.inlayer.inout))

	//fmt.Println("4")
	for i := range wc.weight {
		for j := 0; j < len(inDer); j++ {
			//fmt.Printf("%v, %v | ", i, j)
			inDer[j] += wc.weight[i][j] * weightMod[i][j]
			//fmt.Println("inDer[", j,"]: ", inDer[j], ", t",i,j,": ", weightMod[i][j])
		}
	}

	return inDer
}

//addModeValues
func (wc *weightedConnect) addModValues(weightMod [][]float64, biasMod []float64) {
	//fmt.Println()
	for i := range wc.weight {
		for j := range wc.weight[i] {
			wc.weight[i][j] -= weightMod[i][j] * 0.1
		}
	}
	//fmt.Println("wm: ", weightMod)
	//fmt.Println()

	for i := range wc.bias {
		wc.bias[i] -= biasMod[i] * 0.1
	}
}

// Network builds up the layers of the network
//then connects all the layers
type Network struct {
	Lays       []layer
	Connectors []weightedConnect
}

func (n *Network) ctor() {
	n.Lays = make([]layer, 0, 5)
	n.Connectors = make([]weightedConnect, 0, 5)
	fmt.Println("So I don't have to take fmt out of imports")
}

// AddLayer adds a layer of nodes and a weight connection
// when layer gets to be one more than weight connections
func (n *Network) AddLayer(a act.Activation, nodeCount uint) {
	n.Lays = append(n.Lays, layer{a, make([]float64, nodeCount)})
}

// ConnectLayers Makes all the weights based on the layers in the network
func (n *Network) ConnectLayers() {
	n.Connectors = make([]weightedConnect, len(n.Lays)-1)
	for i := 0; i < len(n.Lays)-1; i++ {
		n.Connectors[i].inlayer = &n.Lays[i]
		n.Connectors[i].outlayer = &n.Lays[i+1]
		n.Connectors[i].init()
	}
}

// AddNode increases the node count of specific
func (n *Network) AddNode(layer int) {
	n.Lays[layer].addNode() // .inout = append(n.Lays[layer].inout, 0.0)
	// the subtracting of a node is going to be all in the weighted connections
	// n.Lays[layer].inout = n.Lays[layer].inout[0:len(n.Lays[layer].inout)]
	n.Connectors[layer-1].outlayer = &n.Lays[layer]
	n.Connectors[layer].inlayer = &n.Lays[layer]
	// add a row to the outlayer
	n.Connectors[layer-1].nodeAdded(true, false)
	// add a column to the inlayer
	n.Connectors[layer].nodeAdded(false, true)
}

// Fillit provides random values, [0.0, 1.0), to input layer
func (n *Network) Fillit() {
	for i := range n.Lays[0].inout {
		n.Lays[0].inout[i] = rand.Float64()
	}
}

// PutData Fills input Layer with provided data
func (n *Network) PutData(input []float64) {
	for i := 0; i < len(n.Lays[0].inout); i++ {
		n.Lays[0].inout[i] = input[i]
	}
}

// GetFinal returns the final layer of values
func (n *Network) GetFinal() []float64 {
	guess := make([]float64, len(n.Lays[len(n.Lays)-1].inout))
	for i := range guess {
		guess[i] = n.Lays[len(n.Lays)-1].inout[i]
	}

	return guess
}

// Propagation passes along the modified data
func (n *Network) Propagation() {
	for _, v := range n.Connectors {
		v.propagate()
	}
}

// BackPropagation is the training function that modifies the weights and biases
func (n *Network) BackPropagation(expected []float64) {
	guess := n.GetFinal()
	err := error(guess, expected)
	//cost := costSqErr(err)
	terr := make([]float64, len(err))

	terr = n.Connectors[len(n.Connectors)-1].outlayer.style.Derivative(guess)

	for i, v := range err {
		terr[i] *= 2. * v
	}

	for i := len(n.Connectors) - 1; i >= 0; i-- {
		terr = n.Connectors[i].brakagate(terr)
		der := n.Connectors[i].inlayer.style.Derivative(n.Connectors[i].inlayer.inout)
		for j := range terr {
			terr[j] *= der[j]
		}
	}
}

func costSqErr(err []float64) []float64 {
	for i, v := range err {
		err[i] = v * v
	}
	return err
}

func error(gs, ex []float64) []float64 {
	ret := make([]float64, len(ex))
	for i := 0; i < len(ex); i++ {
		ret[i] = gs[i] - ex[i]
	}
	return ret
}
