
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class LRG {
	
	public double r;
	public double c;
	public double e;
	public int f;
	
	LRG(double r, double c, double e, int f){
		this.r = r;
		this.c = c;
		this.e = e;
		this.f = f;
	}
	
	public Result run(ArrayList<HashMap<Integer, Double>> data, Double[] w){		
	    Result result = new Result(w, 0, 0, this.r, this.c);
	    Random rng = new Random(12); //Keep shuffling consistent
	    //For each epoch
	    for(int j = 0; j < this.e; j++){
	    	double rate = this.r/(1.0+j);
	    	HashMap<Integer, Double> ex = data.get(rng.nextInt(data.size()));
	    	double value = predict(w, ex);  			
	    	if(Math.max(0, 1-value) == 0){
	    		for(int i = 0; i < w.length-1; i++)
	    			w[i] -= rate*w[i];	    			
	    		w[w.length-1] = w[w.length-1] - rate*w[w.length-1];
	    	}else{
	    		for(int i = 0; i < w.length-1; i++){
	    			if(ex.get(i) != null) 
	    				w[i] -= rate*(w[i]-this.c*ex.get(-1)*ex.get(i));
	    			else 
	    				w[i] -= rate*w[i];
	    		}
	    		w[w.length-1] = w[w.length-1] - rate*(w[w.length-1]-this.c*ex.get(-1));
	    	}
	    	//Keep track of the most accurate epoch so far.
	    	result = new Result(w.clone(), 0, 0, this.r, this.c);
	    	
	    }
		return result;
	}
	
	public double predict(Double[] w, HashMap<Integer, Double>example){
		double prediction = 0;
		for(int i = 0; i < w.length-1; i++){
			if(example.get(i) != null)
				prediction += w[i]*example.get(i);
		}
		prediction += w[w.length-1];
		return prediction * example.get(-1);
	}
	
	public double test(ArrayList<HashMap<Integer, Double>> testset, Double[] w){
		double mistakes = 0;
		for(int i = 0; i < testset.size(); i++){
			HashMap<Integer, Double> ex = testset.get(i);
			double prediction = predict(w, ex);
			if(Math.max(0, 1-prediction) != 0)
				mistakes++;
		}
		return 1-(mistakes/testset.size());
	}
	
	public int[] eval(ArrayList<HashMap<Integer, Double>> evalset, Double[] w){
		int[] predictions = new int[evalset.size()];
		for(int i = 0; i < evalset.size(); i++){
			double label = predict(w, evalset.get(i));
			predictions[i] = 1-label > 1 ? 0:1;
		}
		return predictions;
	}
}
