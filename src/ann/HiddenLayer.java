package ann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HiddenLayer extends Layer {
	
	public HiddenLayer(int numRealUnits, Layer previousLayer) {
		int weightsPerUnit = previousLayer.numUnitsInclBias();
		units = new Unit[numRealUnits + 1];
		super.initWithModelUnit(new HiddenUnit(weightsPerUnit));
	}
	
	public void addUnit(int numWeights) {
		Unit newUnit = new HiddenUnit(numWeights);
		List<Unit> unitList = new ArrayList<Unit>(Arrays.asList(units));
		unitList.add(units.length-1, newUnit);
		units = unitList.toArray(units);
	}

	@Override
	public int numRealUnits() {
		return numUnitsInclBias() - 1;
	}
}
