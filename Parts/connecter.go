package Parts

import "math/rand"

//WeightedConnect provides access to consecutive layers
type WeightedConnect struct {
	Inlayer  *Layer // Getting an "undeclared name: Layer" compiler error - seems wrong
	Outlayer *Layer
	Weight   [][]float64
	Bias     []float64
}

//Init
//inlayer | Weights | outlayer
//   [out][in] * [in] + [out] > I don't know what I was trying to describe
//First  indicies represent the outlayer
//Second indicies represent the inlayer
func (wc *WeightedConnect) Init() {
	wc.Weight = make([][]float64, len(wc.Outlayer.Inout))

	for i := range wc.Weight {
		wc.Weight[i] = make([]float64, wc.Inlayer.Inout.GetCount())
	}

	wc.Bias = make([]float64, wc.Outlayer.Inout.GetCount())

	for i := range wc.Weight {
		for j := range wc.Weight[i] {
			wc.Weight[i][j] = rand.Float64()*2. - 1.
		}
		wc.Bias[i] = rand.Float64()
	}
}

// NodeAdded increases the Weight and bias slices to accomadate the node extention
func (wc *WeightedConnect) nodeAdded(outLayerIncreased bool, inLayerIncreased bool) {
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

//NodeRemoved decreases the weight slice to accomodate the
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
