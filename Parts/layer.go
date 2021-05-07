package Parts

import (
	"errors"

	"github.com/Go-ANN/act"
)

//Layer pairs up an activation functional struct with a slice of float64
type Layer struct {
	Style act.Activation
	Inout []float64
}

//SetInout error if lengths are missmatched
func (l *Layer) SetInout(ace []float64) error {
	if len(ace) != len(l.Inout) {
		return errors.New("Missmatched lengths")
	}
	for i := 0; i < len(ace); i++ {
		l.Inout[i] = ace[i]
	}
	return nil
}

//GetCount same as len(Layer.Inout)
func (l *Layer) GetCount() int {
	return len(l.Inout)
}

//GetInout returns float64 [] of Inout
func (l *Layer) GetInout() []float64 {
	return l.Inout
}

func (l *Layer) AddNode() {
	l.Inout = append(l.Inout, 0.0)
}

//RemoveNode decreases length of Layer.Inout by one
func (l *Layer) RemoveNode() {
	l.Inout = l.Inout[1:]
}
