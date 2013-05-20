package ann;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class SimsThesis {

	public static void main(String[] args) throws IOException {
		double[][] inputsIris = ANN.readPattern("src/ann/irisinputs.txt", "\\s\\s");
		double[][] outputsIris = ANN.readPattern("src/ann/irisoutputs.txt", "\\s\\s");

		ANN irisann = new ANN(1, 0.5, 0.5, 4, 3, 3);
		int basePopSize = 5;
		int serialPopSize = 10;
		double maxError = 0.1;
		int bufferSize = 0;
		
		
		noRehearsalSerialLearning("thesis/norehearsal.txt", irisann, inputsIris, outputsIris, basePopSize, serialPopSize, maxError);
		//fullRehearsalSerialLearning("thesis/fullrehearsal.txt", irisann, inputsIris, outputsIris, basePopSize, serialPopSize, maxError);
		//randomRehearsalSerialLearning("thesis/dRRiris.txt", irisann, inputsIris, outputsIris, 
		//		basePopSize, serialPopSize, maxError, bufferSize);
		//sweepPseudoRehearsalSerialLearning("thesis/dSPRiris.txt", irisann, inputsIris, outputsIris, 
			//	basePopSize, serialPopSize, maxError, bufferSize);
	}
	
	public static void sweepPseudoRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError, int bufferSize) throws IOException {
		SimsA01.createRandomDataSet("src/ann/custominputs.txt", "src/ann/customoutputs.txt", 500);
		inputs = ANN.readPattern("src/ann/custominputs.txt", "\\s\\s");
		outputs = ANN.readPattern("src/ann/customoutputs.txt", "\\s\\s");
		
		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			
			SimsA01.createRandomDataSet("src/ann/custominputs.txt", "src/ann/customoutputs.txt", 500);
			inputs = ANN.readPattern("src/ann/custominputs.txt", "\\s\\s");
			outputs = ANN.readPattern("src/ann/customoutputs.txt", "\\s\\s");
			basein = new double[basePopSize][inputs[0].length];
			baseout = new double[basePopSize][outputs[0].length];
			serialin = new double[serialPopSize][inputs[0].length];
			serialout = new double[serialPopSize][outputs[0].length];
			
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.sweepPseudoRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError, bufferSize);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}
	
	public static void randomRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError, int bufferSize) throws IOException {
		
		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];


		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.randomRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError, bufferSize);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}
	
	public static void fullRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError) throws IOException {
		
		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];


		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		for (int rep = 0; rep < reps; rep++) {
			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

			double[] trial = ann.fullRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}

	public static void noRehearsalSerialLearning(String filename, ANN ann,  double[][] inputs, double[][] outputs,
			int basePopSize, int serialPopSize, double maxError) throws IOException {
		
		double[][] basein = new double[basePopSize][inputs[0].length];
		double[][] baseout = new double[basePopSize][outputs[0].length];
		double[][] serialin = new double[serialPopSize][inputs[0].length];
		double[][] serialout = new double[serialPopSize][outputs[0].length];

		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		int reps = 50;
		double[][] allout = new double[serialin.length+1][reps];

		//createRandomData(inputs, outputs);
		
		//ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);
		
		for (int rep = 0; rep < reps; rep++) {
			
			//createRandomData(inputs, outputs);
			
			//ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);
			//ann.reset();
			// oh hai there gittttt
			double[] trial = ann.noRehearsalSerialLearning(basein, baseout, serialin, serialout, maxError);
			for (int j = 0; j < allout.length; j++) {
				allout[j][rep] = trial[j];
			}
		}

		for (int i = 0; i < allout.length; i++) {
			f.write(i + "\t" + SimsA01.median(allout[i]) + "\n");
		}

		f.close();

	}

	public static void createRandomData(double[][] inputs, double[][] outputs) {
		Random r = new Random(); 
		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[i].length; j++) {
				inputs[i][j] = r.nextDouble()*0.6 - 0.3;
			}
			for (int j = 0; j < outputs[i].length; j++) {
				outputs[i][j] = r.nextDouble()*0.6 - 0.3;
			}
		}
	}
	

}
