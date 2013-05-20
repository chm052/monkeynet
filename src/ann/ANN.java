package ann;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
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


	// probably a bad idea to handle file writing, instead output a double array, where each double
	// is the population error after training another pattern
	// file writing can be handled by a second sims class, much easier
	// length of the output will be equal to length of the @param serialinputs/serialoutputs arrays (+1 for initial error)
	// (their first-level arrays i.e. rows, not their nested arrays)
	public double[] noRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror) {

		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;

		// train base population
		while (maxerror < error) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;

		for (int i = 0; i < serialinputs.length; i++) {
			Pattern trial = new Pattern(serialinputs[i], serialoutputs[i]);

			error = maxerror + 1;
			while (maxerror < error) {
				pass(trial);
				updateAllWeights(trial);
				error = patternError(getCurrentOutput(), serialoutputs[i]);
				System.err.println(error);
			}
			System.err.println();

			double poperror = populationError(outputs(baseinputs), baseoutputs);
			outputs[i+1] = poperror;
		}
		
		return outputs;
	}
	

	public double[] fullRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror) {

		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;

		// train base population
		while (maxerror < error) {
			for (int i = 0; i < baseinputs.length; i++) {
				Pattern trial = new Pattern(baseinputs[i], baseoutputs[i]);
				pass(trial);
				updateAllWeights(trial);
			}
			error = populationError(outputs(baseinputs), baseoutputs);
		}
		outputs[0] = error;

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
			
			while (maxerror < error) {
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

	public double[] randomRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror, int bufferSize) {
		Random r = new Random();
		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;

		// train base population
		while (maxerror < error) {
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
			
			while (maxerror < error) {
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
	
	public double[] sweepPseudoRehearsalSerialLearning(double[][] baseinputs, double[][] baseoutputs, 
			double[][] serialinputs, double[][] serialoutputs, double maxerror, int bufferSize) {
		Random r = new Random();
		double[] outputs = new double[serialinputs.length + 1];	

		double error = maxerror + 1;

		// train base population
		while (maxerror < error) {
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
			rehearseinputs[rehearseinputs.length-1] = serialinputs[i];
			rehearseoutputs[rehearseoutputs.length-1] = serialoutputs[i];

			error = maxerror + 1;
			
			while (maxerror < error) {
				
				pseudopop(rehearseinputs, rehearseoutputs, rehearseinputs.length-1);
				
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

	public void pseudopop(double[][] inputs, double[][] outputs, int size) {
		Random r = new Random();
		
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				inputs[i][j] = r.nextDouble();
			}
			
			pass(new Pattern(inputs[i], null));
			outputs[i] = getCurrentOutput();
		}
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

	public static double populationError(double[][] actual, double[][] expected) {
		double sum = 0;
		for (int i = 0; i < actual.length; i++) {
			sum += patternError(actual[i], expected[i]);
		}
		return sum/actual.length;
	}

	/**
    public boolean isTrained() {
	return getTotalError() < maxError;
    }

    public double getTotalError() {
	double totalError = 0;
	for (int i = 1; i < layers.length; i++) {
	    totalError += layers[i].getError();
	}
	return totalError;
    }

    public void printWeights() {
	//	for (int i = 0; i < layers.size() - 1; i++) {
	//	    System.out.println("LAYER 1\n" + layers.get(i).printWeights());
	//}
	}
	 */

}