package ann;

public class HiddenLayer extends Layer {
	
	public HiddenLayer(int numRealUnits, Layer previousLayer) {
		int weightsPerUnit = previousLayer.numUnitsInclBias();
		units = new Unit[numRealUnits + 1];
		super.initWithModelUnit(new HiddenUnit(weightsPerUnit));
	}

	@Override
	public int numRealUnits() {
		return numUnitsInclBias() - 1;
	}
}
