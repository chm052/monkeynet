package ann;

public class InputLayer extends Layer {

	public InputLayer(int numRealUnits) {
		units = new Unit[numRealUnits + 1];
		initWithModelUnit(new InputUnit());
	}

	@Override
	public int numRealUnits() {
		return numUnitsInclBias() - 1;
	}
}
