package main

import (
	"encoding/json"
	"fmt"
	"os"

	goann "github.com/Go-ANN"
	"github.com/Go-ANN/act"
)

func main() {
	var filepath string
	var filename string
	filepath = "C:\\users\\us3rs\\Desktop\\"
	filename = "testakufdgb.json"

	var x goann.Network

	x.AddLayer(act.Linear{}, 3)
	x.AddLayer(act.ReLU{}, 5)
	x.AddLayer(act.SoftSign{}, 3)
	x.AddLayer(act.Sigmoid{}, 8)
	x.AddLayer(act.ReLU{}, 2)
	x.ConnectLayers()

	fmt.Println(x.SaveJSON(filepath, filename))

	byts, err := json.Marshal(x)
	if err != nil {
		err.Error()
	}
	file, err := os.Create(filepath + "mashal.json")
	file.Write(byts)
	defer file.Close()
}
