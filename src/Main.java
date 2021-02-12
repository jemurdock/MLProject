

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;

public class Main {

	public static void main(String[] args){
		
		BufferedReader reader;
		ArrayList<HashMap<Integer, Double>> fold1 = new ArrayList<>();
		ArrayList<HashMap<Integer, Double>> fold2 = new ArrayList<>();
		ArrayList<HashMap<Integer, Double>> fold3 = new ArrayList<>();
		ArrayList<HashMap<Integer, Double>> fold4 = new ArrayList<>();
		ArrayList<HashMap<Integer, Double>> fold5 = new ArrayList<>();
		ArrayList<HashMap<Integer, Double>> full = new ArrayList<>();
		
		try {
			//Read in training data
			String line;	
			reader = new BufferedReader(new FileReader("data/bow/bow.train.libsvm"));
			line = reader.readLine();
			String[] rawVals = line.split(" ");
			int features = rawVals.length-1; //First column is the label
			int fold = 1;
			while (line != null) {
				HashMap<Integer, Double> example = new HashMap<>();
				rawVals = line.split(" ");
				if(Double.parseDouble(rawVals[0]) == 0.0) //label
					example.put(-1, -1.0);
				else
					example.put(-1, 1.0);								
				for(int i = 1; i < rawVals.length; i++){ //features
					String[] pair = rawVals[i].split(":");
					example.put(Integer.parseInt(pair[0]), Double.parseDouble(pair[1]));
				}
				if(fold%5==1) fold1.add(example);
				else if(fold%5==2) fold2.add(example);
				else if(fold%5==3) fold3.add(example);
				else if(fold%5==4) fold4.add(example);
				else fold5.add(example);
				fold++;
				full.add(example);
				line = reader.readLine();
			}
			reader.close();

			//Run CV
			Double[] sgdR = { 1.0, 0.1, 0.01 };
			Double[] sgdC = { 1000.0, 100.0, 1.0, 0.1 };
			Double[] lrgR = { 1.0, 0.1, 0.01 };
			Double[] lrgC = { 10000.0, 1000.0, 100.0, 1.0, 0.1 };
			SGD sgd = new SGD(sgdR[0], sgdC[0], 10, features);
			LRG lrg = new LRG(lrgR[0], lrgC[0], 200, features);

			Result sgdBest = new Result(null, 0, 0, 0, 0);
			Result lrgBest = new Result(null, 0, 0, 0, 0);
			Double sgdAverage = 0.0;
			Double lrgAverage = 0.0;
			Double count = 0.0;
			Double lrgCount = 0.0;
			for(int i = 0; i < 5; i++){
				ArrayList<HashMap<Integer, Double>> train = new ArrayList<>();
				ArrayList<HashMap<Integer, Double>> eval = new ArrayList<>();
				if(i!=0) train.addAll(fold1); else eval.addAll(fold1);
				if(i!=1) train.addAll(fold2); else eval.addAll(fold2);
				if(i!=2) train.addAll(fold3); else eval.addAll(fold3);
				if(i!=3) train.addAll(fold4); else eval.addAll(fold4);
				if(i!=4) train.addAll(fold5); else eval.addAll(fold5);
				
				//Train and test SGD
				for(int j = 0; j < sgdR.length; j++){
					sgd.r = sgdR[j];
					for(int k = 0; k < sgdC.length; k++){
						sgd.c = sgdC[k];
						//Fold b into weights
					    Double[] w = new Double[features+1];
					    for(int ind = 0; ind < w.length; ind++) w[ind] = 0.0;
						Result r = sgd.run(train, w);
						
						r.testacc = sgd.test(eval, r.weights);
						sgdAverage += r.testacc;
						count++;
						if(r.testacc > sgdBest.testacc)
							sgdBest = r;				
					}
				}
				//Train and test LRG
				for(int l = 0; l < lrgR.length; l++){
					lrg.r = lrgR[l];
					for(int m = 0; m < lrgC.length; m++){
						lrg.c = lrgC[m];
						//Fold b into weights
					    Double[] w = new Double[features+1];
					    for(int ind = 0; ind < w.length; ind++) w[ind] = 0.0;
						Result r = lrg.run(train, w);
						
						r.testacc = lrg.test(eval, r.weights);
						lrgAverage += r.testacc;
						lrgCount++;
						if(r.testacc > lrgBest.testacc)
							lrgBest = r;	
					}
				}
			}

			System.out.println("---------SVM Stochastic Sub-Gradient Descent---------");
			System.out.println("Average Cross-Validation Accuracy: "+sgdAverage/count);
			System.out.println("Best Hyperparameters: learning rate="+sgdBest.rate+", C="+sgdBest.c);
			System.out.println("Best Cross-Validation Accuracy: "+sgdBest.testacc);			
			sgd.r = sgdBest.rate;
			sgd.c = sgdBest.c;   
			Result result = sgd.run(full, sgdBest.weights);
			
			ArrayList<HashMap<Integer, Double>> evalset = new ArrayList<>();
			reader = new BufferedReader(new FileReader("data/bow/bow.eval.anon.libsvm"));
			line = reader.readLine();
			line = reader.readLine(); // Skip the first line with the headers
			while (line != null) {
				HashMap<Integer, Double> example = new HashMap<>();
				example.put(-1, 1.0);
				rawVals = line.split(" ");
				for(int i = 1; i < rawVals.length; i++){
					String[] pair = rawVals[i].split(":");
					example.put(Integer.parseInt(pair[0]), Double.parseDouble(pair[1]));
				}
				evalset.add(example);
				line = reader.readLine();
			}
			reader.close();
			int[] predictions = sgd.eval(evalset, result.weights);			
			//And write the results to the evaluation file.
			File out = new File("data/results.csv");
			FileOutputStream fos;
			fos = new FileOutputStream(out);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));	
			bw.write("example_id,label");
			bw.newLine();
			for(int i = 0; i < predictions.length; i++){
				bw.write(i+","+predictions[i]);
				bw.newLine();
			}
			bw.close();	
			
			System.out.println("----------------Logistic Regression----------------");
			System.out.println("Average Cross-Validation Accuracy: "+lrgAverage/lrgCount);
			System.out.println("Best Hyperparameters: learning rate="+lrgBest.rate+", C="+lrgBest.c);
			System.out.println("Best Cross-Validation Accuracy: "+lrgBest.testacc);
			lrg.r = lrgBest.rate;
			lrg.c = lrgBest.c;    
			result = lrg.run(full, lrgBest.weights);
			//Result result = lrg.run(full, lrgBest.weights);
			

			predictions = lrg.eval(evalset, result.weights);		
			//int[] predictions = lrg.eval(evalset, result.weights);
			//And write the results to the evaluation file.
			out = new File("data/results2.csv");
			fos = new FileOutputStream(out);
			bw = new BufferedWriter(new OutputStreamWriter(fos));	
//			File out = new File("data/results2.csv");
//			FileOutputStream fos = new FileOutputStream(out);
//			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));	
			bw.write("example_id,label");
			bw.newLine();
			for(int i = 0; i < predictions.length; i++){
				bw.write(i+","+predictions[i]);
				bw.newLine();
			}
			bw.close();		
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
