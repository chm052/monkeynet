package ann;


public class HiddenUnit extends Unit {

	public HiddenUnit(int numWeights) {
		init(Unit.randomWeights(numWeights));
	}

	@Override
	public double activationFunction(double rawinput) {
		return 1.0/(1 + Math.exp((-1) * rawinput));
	}
	
	
	@Override
	public Unit clone() {
		return new HiddenUnit(numWeights());
	}
	
}
