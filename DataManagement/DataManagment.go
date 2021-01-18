// Package dtm is here for data and training management
package dtm

import (
	"errors"

	goann "github.com/Go-ANN"
)

// DataBuilder will be a central location for training and testing the network
type DataBuilder struct {
	numInput  uint
	numOutput uint
	xtrain    []float64
	ytrain    []float64
	xtest     []float64
	ytest     []float64
}

// Data is a central location for training and testing the network
type Data struct {
	numInput  uint
	numOutput uint
	xtrain    []float64
	ytrain    []float64
	xtest     []float64
	ytest     []float64
}

// SetNumInput sets the number of input values to the network
func (db *DataBuilder) SetNumInput(in uint) error {
	if len(db.xtest) != 0 || len(db.xtrain) != 0 {
		return errors.New("Already set the db.numInput")
	}
	db.numInput = in
	return nil
}

// SetNumOutput sets the number of output values to the network
func (db *DataBuilder) SetNumOutput(out uint) error {
	if len(db.ytrain) != 0 || len(db.ytest) != 0 {
		return errors.New("Already set the db.numOutput")
	}
	db.numOutput = out
	return nil
}

// AddxTrainValues will add the parameter xTrain values to the db.xtrain slice
// if db.numInput is set and the length of xTrain mod db.numInput is 0
func (db *DataBuilder) AddxTrainValues(xTrain []float64) error {
	if db.numInput == 0 {
		return errors.New("Use SetNumInput(int) before using this function")
	}
	if len(xTrain)%int(db.numInput) != 0 {
		return errors.New("new data to be added is not a multiple of the number of inputs")
	}

	if db.xtrain == nil {
		db.xtrain = make([]float64, 0, db.numInput)
	}

	db.xtrain = append(db.xtrain, xTrain...)

	return nil
}

// AddyTrainValues will add the parameter yTrain values to the db.ytrain slice
// if db.numOutput is set and the length of yTrain mod db.numOutput is 0
func (db *DataBuilder) AddyTrainValues(yTrain []float64) error {
	if db.numOutput != 0 {
		return errors.New("Use SetNumOutput(int) before using this function")
	}
	if len(yTrain)%int(db.numOutput) != 0 {
		return errors.New("new data to be added is not a multiple of the number of Outputs")
	}

	if db.ytrain == nil {
		db.ytrain = make([]float64, 0, db.numOutput)
	}

	db.ytrain = append(db.ytrain, yTrain...)

	return nil
}

// AddxTestValues will add the parameter xTest values to the db.xtest slice
// if db.numInput is set and the length of xTest mod db.numInput is 0
func (db *DataBuilder) AddxTestValues(xTest []float64) error {
	if db.numInput != 0 {
		return errors.New("Use SetNumInput(int) before using this function")
	}
	if len(xTest)%int(db.numInput) != 0 {
		return errors.New("new data to be added is not a multiple of the number of inputs")
	}

	if db.xtest == nil {
		db.xtest = make([]float64, 0, db.numInput)
	}

	db.xtest = append(db.xtest, xTest...)

	return nil
}

// AddyTestValues will add the parameter yTest values to the db.ytest slice
// if db.numOutput is set and the length of yTest mod db.numOutput is 0
func (db *DataBuilder) AddyTestValues(yTest []float64) error {
	if db.numOutput == 0 {
		return errors.New("Use SetNumOutput(int) before using this function")
	}
	if len(yTest)%int(db.numOutput) != 0 {
		return errors.New("new data to be added is not a multiple of the number of Outputs")
	}

	if db.ytest == nil {
		db.ytest = make([]float64, 0, db.numOutput)
	}

	db.ytest = append(db.ytest, yTest...)

	return nil
}

// Build returns the Data struct which will be used to train and test
// Network held separatly
func (db *DataBuilder) Build() Data {
	return Data{db.numInput, db.numOutput, db.xtrain, db.ytrain, db.xtest, db.ytest}
}

// Train corrects the provided Netork to the data held within the Data.
// iterations - length of training loop
// batch - has yet to work
// ann - the provided Network
func (d Data) Train(iterations uint, ann goann.Network) goann.Network {
	for ind, c := 0, 0; c < int(iterations); c++ {
		index1, index2 := (ind*int(d.numInput))%len(d.xtrain), (ind*int(d.numInput)+int(d.numInput))%len(d.xtrain)
		outdex1, outdex2 := (ind*int(d.numOutput))%len(d.ytrain), (ind*int(d.numOutput)+int(d.numOutput))%len(d.ytrain)
		ann.PutData(d.xtrain[index1:index2])
		ann.Propagation()
		//fmt.Println(input, x.GetFinal())
		ann.BackPropagation(d.ytrain[outdex1:outdex2])
		ind++

	}

	return ann
}

// Test will show information on accuracy of test guesses and train guesses.
// Maybe even updating presented info as it progresses through the data.
func (d Data) Test(howmany uint, ann goann.Network) {
	// Nothing right now
}
