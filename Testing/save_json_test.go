package Testing

import (
	"fmt"
	"os"
	"testing"

	goann "github.com/Go-ANN"
	"github.com/Go-ANN/act"
)

func TestSave(t *testing.T) {
	var x goann.Network
	var inout uint = 3

	x.AddLayer(act.Linear{}, inout)
	x.AddLayer(act.Linear{}, 5)
	x.AddLayer(act.Sigmoid{}, 6)
	x.AddLayer(act.SoftSign{}, 4)
	x.AddLayer(act.ReLU{}, 5)
	x.AddLayer(act.ReLU{}, inout)
	x.ConnectLayers()

	var filepath string = "C:\\users\\us3rs\\Desktop\\NetSave\\"
	var filename string = "testakufdgb.json"
	var size int = x.SaveJSON(filepath, filename)
	fmt.Println(x.SaveJSON(filepath, filename))
	fi, err := os.Stat(filepath + filename)

	if err != nil {
		t.Errorf("The file is not even found %s", err.Error())
	}

	if !(fi.Size() > int64((float64(size)*0.95)) && fi.Size() < int64((float64(size)*1.05))) {
		t.Errorf("Seems the size of the file is not near the expected size. (actual : expected) > (%v : %v)", fi.Size(), size)
	}

	// I might be able to delete the file if it is present
	// then check the json.Unmarshal()
	// next time
}
