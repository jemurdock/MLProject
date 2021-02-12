
public class Result{
		
	Double[] weights;
	double accuracy;
	double mistakes;
	double testacc;
	double rate;
	double c; //-1 if c is not used
		
	Result(Double[] w, double a, double m, double r, double c){
		weights = w;
		accuracy = a;
		mistakes = m;
		rate = r;
		this.c  = c;
		testacc = 0;
	}
}
