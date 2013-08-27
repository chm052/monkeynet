package ann;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

import com.google.common.annotations.VisibleForTesting;

public class ANN {
	private final Layer[] layers;
	public double learningSpeed;
	public double momentum;

	public ANN(int numHiddenLayers, double learningSpeed, double momentum,
			int... sizeLayers) {
		int numLayers = numHiddenLayers + 2;

		layers = new Layer[numLayers];
		this.learningSpeed = learningSpeed;
		this.momentum = momentum;

		layers[0] = new InputLayer(sizeLayers[0]);
		for (int i = 1; i <= numHiddenLayers; i++) {
			layers[i] = new HiddenLayer(sizeLayers[i], layers[i-1]);
		}
		layers[numLayers-1] = new OutputLayer(sizeLayers[numLayers-1], 
				layers[numLayers-2]);
	}

	public double[][] outputs(double[][] inputs) {
		double[][] outputs = new double[inputs.length][layerSize(numLayers()-1)];
		for (int j = 0; j < inputs.length; j++) {
			Pattern trial = new Pattern(inputs[j], null);
			pass(trial);
			double[] currentoutput = getCurrentOutput();
			outputs[j] = currentoutput;
		}
		return outputs;
	}

	public void reset() {
		layers[0] = new InputLayer(layers[0].numRealUnits());
		for (int i = 1; i < numLayers()-1; i++) {
			layers[i] = new HiddenLayer(layers[i].numRealUnits(), layers[i-1]);
		}
		layers[numLayers()-1] = new OutputLayer(layers[numLayers()-1].numRealUnits(), 
				layers[numLayers()-2]);
	}

	@Override
	public String toString() {
		String s = "**ANN** " + "\n";
		for (int i = 0; i < numLayers(); i++) {
			s += "Layer " + i + ": " + layers[i] + "\n";
		}
		return s;
	}

	public int numLayers() {
		return layers.length;
	}

	public void pass(Pattern pattern) {
		setAsInput(pattern);
		for (int i = 1; i < numLayers(); i++) {
			layers[i].propagate(layers[i-1]);
		}
	}

	public void setAsInput(Pattern pattern) {
		layers[0].setOutput(pattern);
	}

	public double[] getOutput(int layer) {
		return layers[layer].getOutput();
	}

	public double[] getCurrentOutput() {
		return layers[numLayers() - 1].getOutput();
	}

	public void updateAllWeights(Pattern target) {
		layers[numLayers() - 1].setOuterErrors(target);

		for (int i = numLayers() - 2; i > 0; i--) {
			layers[i].setHiddenErrors(layers[i+1]);
		}

		for (int i = numLayers() - 1; i > 0; i--) {
			layers[i].updateWeights(layers[i-1], learningSpeed, momentum);
		}
	}

	public int layerSize(int layer) {
		return layers[layer].numRealUnits();
	}

	@VisibleForTesting
	public void setLayer(Layer layer, int i) {
		layers[i] = layer;
	}

	// length of the output will be equal to length of the @param serialinputs/serialoutputs arrays (+1 for initial error)
	// (their first-level arrays i.e. rows, not their nested arrays)
	public double[] noRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror) {

		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;
		int counter = 0;	
		// train base population
		while (maxerror < error && counter++ < 1000) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;
		if (counter>=1000) { System.err.println("counter exceeded 1000");}

		for (int i = 0; i < serialinputs.length; i++) {
			Pattern trial = new Pattern(serialinputs[i], serialoutputs[i]);

			error = maxerror + 1;
			counter = 0;
			while (maxerror < error && counter++ < 1000) {
				pass(trial);
				updateAllWeights(trial);
				error = patternError(getCurrentOutput(), serialoutputs[i]);
				//System.err.println(error);
			}
			//System.err.println();

			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		if (counter>=1000) { System.err.println("counter exceeded 1000"); }
		
		return outputs;
	}
	
	public double[] fullRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror) {

		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;
		int counter = 0;
		// train base population
		while (maxerror < error && counter++ < 1000) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;
		if (counter>=1000) { System.err.println("counter exceeded 1000"); }


		for (int i = 0; i < serialinputs.length; i++) {
			
			double[][] rehearseinputs = new double[baseinputs.length + i + 1][];
			double[][] rehearseoutputs = new double[baseinputs.length + i + 1][];
			for (int j = 0; j < baseinputs.length; j++) {
				rehearseinputs[j] = baseinputs[j];
				rehearseoutputs[j] = baseoutputs[j];
			}
			for (int j = 0; j <= i; j++) {
				rehearseinputs[baseinputs.length + j] = serialinputs[j];
				rehearseoutputs[baseinputs.length + j] = serialoutputs[j];
			}

			error = maxerror + 1;
			counter = 0;
			while (maxerror < error && counter++ < 1000) {
				for (int j = 0; j < rehearseinputs.length; j++) {
					Pattern trial = new Pattern(rehearseinputs[j], rehearseoutputs[j]);
					pass(trial);
					updateAllWeights(trial);
				}
				error = populationError(outputs(rehearseinputs), rehearseoutputs);
			}
			if (counter>=1000) { System.err.println("counter exceeded 1000"); }

			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		
		return outputs;
	}
	
	public double[] sweepRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror) {

		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;
		int counter = 0;
		// train base population
		while (maxerror < error && counter++ < 1000) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;
		if (counter>=1000) { System.err.println("counter exceeded 1000"); }  else {
			System.out.println("Learned within " + counter + " epochs.");
		}

// ^^^^ You are here. What's with the js?
		for (int i = 0; i < serialinputs.length; i++) {
			
			double[][] rehearseinputs = new double[baseinputs.length + i + 1][];
			double[][] rehearseoutputs = new double[baseinputs.length + i + 1][];
			for (int j = 0; j < baseinputs.length; j++) {
				rehearseinputs[j] = baseinputs[j];
				rehearseoutputs[j] = baseoutputs[j];
			}
			for (int j = 0; j <= i; j++) {
				rehearseinputs[baseinputs.length + j] = serialinputs[j];
				rehearseoutputs[baseinputs.length + j] = serialoutputs[j];
			}

			error = maxerror + 1;
			counter = 0;
			while (maxerror < error && counter++ < 10000) {
				
				selectpop(rehearseinputs, rehearseoutputs, baseinputs.length, baseinputs, baseoutputs);
				
				for (int j = 0; j < rehearseinputs.length; j++) {
					Pattern trial = new Pattern(rehearseinputs[j], rehearseoutputs[j]);
					pass(trial);
					updateAllWeights(trial);
				}
				error = populationError(outputs(rehearseinputs), rehearseoutputs);
			}
			if (counter>=1000) { System.err.println("counter exceeded 1000"); } else {
				System.out.println("Learned within " + counter + " epochs.");
			}

			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		
		return outputs;
	}

	public double[] randomRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror, int bufferSize) {
		Random r = new Random();
		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;
		int counter = 0;
		// train base population
		while (maxerror < error && counter++ < 1000) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;

		for (int i = 0; i < serialinputs.length; i++) {
			
			double[][] rehearseinputs = new double[bufferSize + 1][];
			double[][] rehearseoutputs = new double[bufferSize + 1][];
			for (int j = 0; j < rehearseinputs.length - 1; j++) {
				int n = r.nextInt(rehearseinputs.length);
				rehearseinputs[j] = baseinputs[n];
				rehearseoutputs[j] = baseoutputs[n];
			}
			rehearseinputs[rehearseinputs.length-1] = serialinputs[i];
			rehearseoutputs[rehearseoutputs.length-1] = serialoutputs[i];

			error = maxerror + 1;
			counter = 0;
			while (maxerror < error && counter++ < 1000) {
				for (int j = 0; j < rehearseinputs.length; j++) {
					Pattern trial = new Pattern(rehearseinputs[j], rehearseoutputs[j]);
					pass(trial);
					updateAllWeights(trial);
				}
				error = populationError(outputs(rehearseinputs), rehearseoutputs);
			}

			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		
		return outputs;
	}
	
	public void makepseudopop(double[][] inputs, double[][] outputs, boolean reals, double minrange, double maxrange) {
		Random r = new Random();
		
		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				if (reals) {
					inputs[i][j] = r.nextDouble() * (maxrange - minrange) + minrange;
				} else {
					inputs[i][j] = r.nextDouble() < 0.5 ? 0 : 1;
				}
			}
			
			pass(new Pattern(inputs[i], null));
			outputs[i] = getCurrentOutput();
		}
		
	}
	
	public double[] sweepPseudoRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror, int bufferSize, boolean reals, double minrange, double maxrange) {
		double[] outputs = new double[serialinputs.length + 1];	
		//System.out.println("Entering sweep pseudorehearsal");
		double error = maxerror + 1;
		
		// train base population
		int cutoff = 0;
		while (maxerror < error && cutoff++ < 100000) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;
		//System.out.println("Trained on base pop");
		if (cutoff>=1000) { System.err.println("counter exceeded 1000"); }  else {
			//System.out.println("Learned within " + cutoff + " epochs.");
		}
		
		double[][] pseudoinputs = new double[10][baseinputs[0].length];
		double[][] pseudooutputs = new double[10][baseoutputs[0].length];
		
		makepseudopop(pseudoinputs, pseudooutputs, reals, minrange, maxrange);
		
		/*System.out.println("PSEUDOPOP INPUTS");
		for (int i = 0; i < pseudoinputs.length; i++) {
			System.out.println(Arrays.toString(pseudoinputs[i]));
		}
		System.out.println();*/
		//System.out.println("\tONE: made pseudopop:\n" + Arrays.deepToString(pseudoinputs) + "\n" + Arrays.deepToString(pseudooutputs));

		for (int i = 0; i < serialinputs.length; i++) {
			//System.out.println("\tTWO: training serial input " + i);
			double[][] rehearseinputs = new double[bufferSize + 1][serialinputs[i].length];
			double[][] rehearseoutputs = new double[bufferSize + 1][serialoutputs[i].length];
			rehearseinputs[rehearseinputs.length-1] = serialinputs[i];
			rehearseoutputs[rehearseoutputs.length-1] = serialoutputs[i];

			error = maxerror + 1;
			cutoff = 0;
			maxerror *= 1.5;
			while (maxerror < error && cutoff++ < 100000) {
				
				selectpop(rehearseinputs, rehearseoutputs, rehearseinputs.length-1, pseudoinputs, pseudooutputs);
				
				for (int j = 0; j < rehearseinputs.length; j++) {
					Pattern trial = new Pattern(rehearseinputs[j], rehearseoutputs[j]);
					pass(trial);
					updateAllWeights(trial);
				}
				//pass(rehearseinputs[rehearseinputs.length-1]); // CHECK!
				error = patternError(getCurrentOutput(), 
						rehearseoutputs[rehearseoutputs.length-1]); //CHANGED!!! CHANGED!!!!
				//System.out.println("THREE: error " + error);
			}
			//System.out.println("\tTHREE: trained "  + i);
			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		if (cutoff>=1000) { System.err.println("counter exceeded 1000"); }  else {
			//System.out.println("Learned within " + cutoff + " epochs.");
		}
		//System.out.println("DONE: error " + populationError(outputs(baseinputs), baseoutputs));
		//System.out.println("DONE: error " + populationError(outputs(serialinputs), serialoutputs));

		//System.out.println("\n"+this+"\n");
		return outputs;
	}

	public static void randomTrainPopulation(double[][] inputs, double[][] trainin, double[][] testin,
			double[][] outputs, double[][] trainout, double[][] testout) {

		ArrayList<Integer> selected = new ArrayList<Integer>();
		for (int i = 0; i < inputs.length; i++) selected.add(i);
		Collections.shuffle(selected);

		for (int i = 0; i < trainin.length; i++) {
			trainin[i] = inputs[selected.get(i)];
			trainout[i] = outputs[selected.get(i)];
		}

		for (int i = trainin.length; i < trainin.length + testin.length; i++) {
			testin[i-trainin.length] = inputs[selected.get(i)];
			testout[i-trainin.length] = outputs[selected.get(i)];
		}
	}

	public void selectpop(double[][] inputs, double[][] outputs, int size, double[][] inputpool, double[][] outputpool) {
		//System.out.println("size: " + size + ", max: " + inputpool.length + " " + outputpool.length + " " + inputs.length + " " + outputs.length);
		permute(inputpool, outputpool);
		
		for (int i = 0; i < size; i++) {
			inputs[i] = inputpool[i];
			outputs[i] = outputpool[i];
		}
		//System.out.println("!:\n" + Arrays.deepToString(inputs) + "\n" + Arrays.deepToString(outputs));
	}

	 public static double[][] readPattern(String filename, String separator) throws FileNotFoundException {

		File file = new File(filename);
		Scanner s = new Scanner(file);
		Scanner ls;
		ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
		String line;
		ArrayList<Double> next;
		while (s.hasNextLine()) {

			line = s.nextLine();
			ls = new Scanner(line);
			next = new ArrayList<Double>();
			while (ls.hasNextDouble()) {
				next.add(ls.nextDouble());
			}
			temp.add(next);
			ls.close();
		}

		double[][] out = new double[temp.size()][temp.get(0).size()];
		for (int i = 0; i < out.length; i++) {
			for (int j = 0; j < out[0].length; j++) {
				out[i][j] = temp.get(i).get(j);
			}
		}

		s.close();
		return out;
	}

	public static void permute(double[][] inputs, double[][] outputs) {
		Random r = new Random();

		int j;
		double[] temp;
		for (int i = inputs.length-1; i >= 1; i--) {
			j = r.nextInt(i + 1);

			temp = inputs[j];
			inputs[j] = inputs[i];
			inputs[i] = temp;

			temp = outputs[j];
			outputs[j] = outputs[i];
			outputs[i] = temp;
		}
	}

	public static double patternError(double[] actualoutput, double[] expectedoutput) {
		double sum = 0;
		for (int i = 0; i < actualoutput.length; i++) {
			sum += 0.5 * Math.pow(actualoutput[i] - expectedoutput[i], 2);
		}
		return sum;
	}

	public static double populationError(double[][] actual, double[][] expected) { // the mean
		double sum = 0;
		for (int i = 0; i < actual.length; i++) {
			sum += patternError(actual[i], expected[i]);
		}
		return sum/actual.length;
	}

	public double[][] dynamicSweepPseudo(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror, int bufferSize, double abortlim, boolean reset,
			boolean reals) {
		double[] outputs = new double[serialinputs.length + 1];	
		System.out.println("Entering sweep pseudorehearsal");
		
		// train base population
		//while (maxerror < error && cutoff++ < 1000) {
		//	for (int i = 0; i < baseinputs.length; i++) {
		//		Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
		//		pass(trial);
		//		updateAllWeights(trial);
		//	}
		//	error = populationError(outputs(baseinputs), baseoutputs);
		//}
		ArrayList<Double> baseErrs = new ArrayList<Double>();
		ArrayList<Double> serialErrs = new ArrayList<Double>();

		dynamicTrain(baseinputs, baseoutputs, maxerror, reset);
		outputs[0] = populationError(outputs(baseinputs), baseoutputs);
		System.out.println("Trained on base pop");
		
		double[][] pseudoinputs = new double[8][baseinputs[0].length]; // FIX LITERAL
		double[][] pseudooutputs = new double[8][baseoutputs[0].length];
		
		makepseudopop(pseudoinputs, pseudooutputs, reals, 0, 1);
		System.out.println("\tONE: made pseudopop:\n" + Arrays.deepToString(pseudoinputs) + "\n" + Arrays.deepToString(pseudooutputs));

		for (int i = 0; i < serialinputs.length; i++) {
			System.out.println("\tTWO: training serial input " + i);

			//error = maxerror + 1;
			//cutoff = 0;
		//	while (maxerror < error && cutoff++ < 1000) {
				
		//		selectpop(rehearseinputs, rehearseoutputs, rehearseinputs.length-1, pseudoinputs, pseudooutputs);
				
		//		for (int j = 0; j < rehearseinputs.length; j++) {
		//			Pattern trial = new Pattern(rehearseinputs[j], rehearseoutputs[j]);
		//			pass(trial);
		//			updateAllWeights(trial);
		//		}
		//		error = populationError(outputs(rehearseinputs), rehearseoutputs);
				//System.out.println("THREE: error " + error);
		//	}
			makepseudopop(pseudoinputs, pseudooutputs, reals, 0, 1);

			double[][] errs = dynamicPseudoTrain(pseudoinputs, pseudooutputs, serialinputs[i], serialoutputs[i], baseinputs, baseoutputs, maxerror, bufferSize, abortlim, reset);
			
			for (int ii = 0; ii < errs[0].length; ii++) {
				baseErrs.add(errs[0][i]);
			}
			for (int ii = 0; ii < errs[0].length; ii++) {
				serialErrs.add(errs[1][i]);
			}
			// NNEEEDS A METTHTHOOODDD
			System.out.println("\tTHREE: trained "  + i);
			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		
		System.out.println("Dynamic training finished. Results:");
		for (int i = 0; i < baseinputs.length; i++) {
			Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
			pass(trial);
			System.out.println(Arrays.toString(baseinputs[i]) + "\t" + Arrays.toString(getCurrentOutput()) + "\n");
		}
		for (int i = 0; i < serialinputs.length; i++) {
			Pattern trial = new Pattern(serialinputs[i], serialoutputs[i]);
			pass(trial);
			System.out.println(Arrays.toString(serialinputs[i]) + "\t" + Arrays.toString(getCurrentOutput()) + "\n");
		}
		double[][] toReturn = new double[2][baseErrs.size()];
		
		for (int i = 0; i < baseErrs.size(); i++) {
			toReturn[0][i] = baseErrs.get(i);
			toReturn[1][i] = serialErrs.get(i);
		}
		
		return toReturn;
	}
	
	public double[][] dynamicPseudoTrain(double[][] pseudoinputs, double[][] pseudooutputs,
			double[] toLearnIn, double[] toLearnOut, double[][] basecompin,
			double[][] basecompout, double maxerror, int bufferSize, double abortlim, boolean reset) {
		double error = maxerror + 1;
		int counter = 0;
		
		double[][] rehearseinputs = new double[bufferSize + 1][toLearnIn.length];
		double[][] rehearseoutputs = new double[bufferSize + 1][toLearnOut.length];
		rehearseinputs[rehearseinputs.length-1] = toLearnIn;
		rehearseoutputs[rehearseoutputs.length-1] = toLearnOut;
		
		ArrayList<Double> baseError = new ArrayList<Double>();
		ArrayList<Double> serialError = new ArrayList<Double>();

		while (maxerror < error) {
			counter++;
			
			if (counter%500==0) {
				serialError.add(error);
				baseError.add(populationError(outputs(basecompin), basecompout));
				//System.out.println("Trial " + counter + ": " + error);
			}
			if (counter >= abortlim) {
				counter = 0;
				System.out.println("Adding hidden unit...");
				for (int i = 0; i < rehearseinputs.length; i++) {
					Pattern trial = new Pattern(rehearseinputs[i], rehearseoutputs[i]);
					pass(trial);
					System.out.println(Arrays.toString(rehearseinputs[i]) + " should be " + Arrays.toString(rehearseoutputs[i]) + " was " + Arrays.toString(getCurrentOutput()) + "\n");
				}
				addHiddenUnit(reset);
				System.out.println("\nADDED UNIT\n----------\n" + this);
			}
			
			selectpop(rehearseinputs, rehearseoutputs, rehearseinputs.length-1, pseudoinputs, pseudooutputs);
			
			for (int i = 0; i < rehearseinputs.length; i++) {
				Pattern trial = new Pattern(rehearseinputs[i], rehearseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			pass(new Pattern(rehearseinputs[rehearseinputs.length-1], rehearseoutputs[rehearseinputs.length-1]));
			error = patternError(getCurrentOutput(),  
					rehearseoutputs[rehearseoutputs.length-1]);
			
		}
		
		pass(new Pattern(rehearseinputs[rehearseinputs.length-1], rehearseoutputs[rehearseinputs.length-1]));
		System.out.println("Trained: " + Arrays.toString(getCurrentOutput()) + " -> " + Arrays.toString(rehearseoutputs[rehearseinputs.length-1]));
		
		double[] be = new double[baseError.size()];
		double[] se = new double[serialError.size()];
		
		for (int i = 0; i < baseError.size(); i++) {
			be[i] = baseError.get(i);
			se[i] = serialError.get(i);
		}
		double[][] out = {be, se};
		return out;
		//System.out.println("Trial " + counter + ": " + error + "\n" + this + "\n\n");
	}
	
	// must be 3-layer network, or least definitely not less than 3 layers.
	public double[] dynamicTrain(double[][] inputs, double[][] outputs, double maxerror, boolean reset) {
		//System.out.println("Welcome to dynamic error training! It's like cascorr but not!");
		//System.out.println("Starting configuration:\n" + this);
		double error = maxerror + 1;
		int counter = 0;
		ArrayList<Double> errs = new ArrayList<Double>();

		while (maxerror < error) {
			counter++;
			if (counter%100==0) errs.add(error);//System.out.println("Trial " + counter + ": " + error);
			if (counter >= 1000) { // MAGIC NUM IS 425 FOR LEARNING XOR WITH ONE HL NODE
				counter = 0;
				addHiddenUnit(reset);
				System.out.println("\nADDED UNIT\n----------\n" + this);
			}
			
			for (int i = 0; i < inputs.length; i++) {
				Pattern trial = new Pattern(inputs[i], outputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(inputs), outputs);
			
			
		}
		//System.out.println("Trial " + counter + ": " + error + "\n" + this + "\n\n");
		
		for (int i = 0; i < inputs.length; i++) {
			Pattern trial = new Pattern(inputs[i], outputs[i]);
			pass(trial);
			System.out.println(Arrays.toString(inputs[i]) + "\t" + Arrays.toString(getCurrentOutput()) + "\n");
		}
		double[] outerr = new double[errs.size()];
		for (int i = 0; i < errs.size(); i++) {
			outerr[i] = errs.get(i);
		}
		return outerr;
	}
	
	public void addHiddenUnit(boolean reset) {
		((HiddenLayer)layers[1]).addUnit(layers[0].numUnitsInclBias()); //baddddddddd
		for (int i = 0; i < layers[2].numRealUnits(); i++) {
			layers[2].unit(i).addRandomWeight();
		}
		if (reset) {
			reset();
		}
	}

}