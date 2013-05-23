package ann;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class SimsThesis {

	public static void main(String[] args) throws IOException {
		double[][] inputsIris = ANN.readPattern("src/ann/irisinputs.txt", "\\s\\s");
		double[][] outputsIris = ANN.readPattern("src/ann/irisoutputs.txt", "\\s\\s");

		double[][] inputsSpiral = ANN.readPattern("src/ann/spiralinput.txt", "\\s\\s");
		double[][] outputsSpiral = ANN.readPattern("src/ann/spiraloutput.txt", "\\s\\s");

		ANN irisann = new ANN(1, 0.5, 0.5, 4, 3, 3);
		ANN spiralann = new ANN(1, 0.5, 0.5, 2, 15, 1);
		
		int basePopSize = 5;
		int serialPopSize = 10;
		double maxError = 0.1;
		int bufferSize = 0;

		double[][] randin = new double[20][4];
		double[][] randout = new double[20][3];
		int[] hlsizes = {1, 10, 25, 50, 100};
		
		
		findBestOverfittingByHL("thesis/spiraloverfitHL.txt", 5, 100000, 2000, spiralann, hlsizes, inputsSpiral, outputsSpiral);
		//findBestHL("thesis/spiralHL.txt", 20, 70000, 2000, spiralann, hlsizes, inputsSpiral, outputsSpiral);
		//trackError("thesis/spiraltrack.txt", 20, 100000, 2000, spiralann, inputsSpiral, outputsSpiral);
		//noRehearsalSerialLearning("thesis/norehearsal.txt", irisann, randin, randout, basePopSize, serialPopSize, maxError);
		//fullRehearsalSerialLearning("thesis/fullrehearsal.txt", irisann, inputsIris, outputsIris, basePopSize, serialPopSize, maxError);
		//randomRehearsalSerialLearning("thesis/dRRiris.txt", irisann, inputsIris, outputsIris, 
		//		basePopSize, serialPopSize, maxError, bufferSize);
		//sweepPseudoRehearsalSerialLearning("thesis/dSPRiris.txt", irisann, inputsIris, outputsIris, 
		//	basePopSize, serialPopSize, maxError, bufferSize);
	}
	
	public static void findBestOverfittingByHL(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, int[] hlsizes, double[][] inputs, double[][] outputs) throws IOException {
		int numDataPoints = epochsPerRep/printFrequency;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());

		double[][] trainin = new double[inputs.length/3][inputs[0].length];
		double[][] trainout = new double[outputs.length/3][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		System.err.println(Arrays.deepToString(trainin) + "\n"
				+ Arrays.deepToString(trainout));

		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int hlsize = 0; hlsize < hlsizes.length; hlsize++) {
			ann = new ANN(1, ann.learningSpeed, ann.momentum, ann.layerSize(0), hlsizes[hlsize], ann.layerSize(2));
			System.err.println(ann.layerSize(1));
			double[][] tracks = new double[numDataPoints][reps];
			for (int q = 0; q < reps; q++) {
				ann.reset();
				
				ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

				int dataPoint = 0;
				for (int i = 0; i < epochsPerRep; i++) {
					ANN.permute(trainin, trainout);
					for (int j = 0; j < trainin.length; j++) {
						Pattern trial = new Pattern(trainin[j], trainout[j]);
						ann.pass(trial);
						ann.updateAllWeights(trial);
					}
					double err = ANN.populationError(ann.outputs(testin), testout);
					if (i%printFrequency == 0) {
						tracks[dataPoint][q] = err;
						dataPoint++;
					}
				}
			}
			f.write(hlsizes[hlsize] + "\t");
			for (int dp = 0; dp < tracks.length; dp++) {
				f.write(SimsA01.median(tracks[dp]) + "\t");
				//System.err.println(median(tracks[dp]));
			}
			f.write("\n");
		}
		f.close();
	}
	
	public static void findBestHL(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, int[] hlsizes, double[][] inputs, double[][] outputs) throws IOException {
		int numDataPoints = epochsPerRep/printFrequency;
		File file = new File(filename);
		FileWriter f = new FileWriter(file.getAbsolutePath());
		/*
		double[][] trainin = new double[inputs.length/3][inputs[0].length];
		double[][] trainout = new double[outputs.length/3][outputs[0].length];
		double[][] testin = new double[inputs.length-trainin.length][inputs[0].length];
		double[][] testout = new double[outputs.length-trainout.length][outputs[0].length];

		ANN.randomTrainPopulation(inputs, trainin, testin, outputs, trainout, testout);

		System.err.println(Arrays.deepToString(trainin) + "\n"
				+ Arrays.deepToString(trainout));*/

		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int hlsize = 0; hlsize < hlsizes.length; hlsize++) {
			ann = new ANN(1, ann.learningSpeed, ann.momentum, ann.layerSize(0), hlsizes[hlsize], ann.layerSize(2));
			System.err.println(ann.layerSize(1));
			double[][] tracks = new double[numDataPoints][reps];
			for (int q = 0; q < reps; q++) {
				ann.reset();
				
				int dataPoint = 0;
				for (int i = 0; i < epochsPerRep; i++) {
					ANN.permute(inputs, outputs);
					for (int j = 0; j < inputs.length; j++) {
						Pattern trial = new Pattern(inputs[j], outputs[j]);
						ann.pass(trial);
						ann.updateAllWeights(trial);
					}
					double err = ANN.populationError(ann.outputs(inputs), outputs);
					if (i%printFrequency == 0) {
						tracks[dataPoint][q] = err;
						dataPoint++;
					}
				}
			}
			f.write(hlsizes[hlsize] + "\t");
			for (int dp = 0; dp < tracks.length; dp++) {
				f.write(SimsA01.median(tracks[dp]) + "\t");
				//System.err.println(dp + "/" + tracks.length + ": " + SimsA01.median(tracks[dp]));
			}
			f.write("\n");
		}
		f.close();
	}
	
	public static void trackError(String filename, int reps, int epochsPerRep, int printFrequency,
			ANN ann, double[][] inputs, double[][] outputs) throws IOException {
		File file = new File(filename); //new File("omg_ann_or.txt");
		FileWriter f = new FileWriter(file.getAbsolutePath());
		f.write("0\t");
		for (int i = 0; i < epochsPerRep; i++) {
			if (i%printFrequency == 0) f.write(i + "\t");
		}
		f.write("\n");
		for (int q = 0; q < reps; q++) {
			f.write(q + "\t");
			ann.reset();
			for (int i = 0; i < epochsPerRep; i++) {
				double[][] currentoutputs = new double[inputs.length][outputs[0].length];
				ANN.permute(inputs, outputs);
				for (int j = 0; j < inputs.length; j++) {
					Pattern trial = new Pattern(inputs[j], outputs[j]);
					ann.pass(trial);
					ann.updateAllWeights(trial);
					double[] currentoutput = ann.getCurrentOutput();
					currentoutputs[j] = currentoutput;
				}
				double err = ANN.populationError(ann.outputs(inputs), outputs);
				if (i%printFrequency == 0) {
					/*f.write(Arrays.deepToString(inputs)+ " " + Arrays.deepToString(ann.outputs(inputs))+"\t");//*/
					f.write(err + "\t");
					System.err.println(i + ": " + err);
				}
			}
			f.write("\n");
		}
		f.close();
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

		int reps = 5;
		double[][] allout = new double[serialin.length+1][reps];

		//createRandomData(inputs, outputs);
		//ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);

		for (int rep = 0; rep < reps; rep++) {

			//createRandomData(inputs, outputs);

			ANN.randomTrainPopulation(inputs, basein, serialin, outputs, baseout, serialout);
			ann.reset();

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
				inputs[i][j] = r.nextDouble();
			}
			for (int j = 0; j < outputs[i].length; j++) {
				outputs[i][j] = r.nextDouble();
			}
		}
	}

}
