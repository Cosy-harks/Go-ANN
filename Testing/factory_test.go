package Testing

import (
	"fmt"
	"testing"

	"github.com/Go-ANN/act"
)

func TestFactory(t *testing.T) {
	nums := []float64{1.1, 2.3, 0.89, -2.}
	name1 := "linear"
	want := "Linear"
	activr := act.ActivationFactory(name1)
	got := activr.ToString()
	if want != got {
		t.Errorf("got = %s, but want = %s", got, want)
	}
	fmt.Printf("%s from act.ActivationFactory(%s).Act(%f) = %f\n", got, name1, nums, activr.Act(nums))

	want = "SoftSign"
	name2 := "softsign"
	activr = act.ActivationFactory(name2)
	got = activr.ToString()
	if want != got {
		t.Errorf("got = %s, but want = %s", got, want)
	}
	fmt.Printf("%s from act.ActivationFactory(%s).Act(%f) = %f\n", got, name2, nums, activr.Act(nums))
}
