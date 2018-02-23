import javax.lang.model.type.ArrayType;
import java.lang.annotation.Target;
import java.util.*;

public class Utils<T extends Number> {

    public static double logSumExp(List<Double> log_x) {

        int N=log_x.size();
        double log_xMax= Collections.max(log_x);
        if(log_xMax==-1*Double.POSITIVE_INFINITY)
            return log_xMax;

        double sum=0;

        for(int i=0; i<N; i++){
            sum+=Math.exp(log_x.get(i)-log_xMax);
        }

        return Math.log(sum)+log_xMax;

    }

    public static double logSumExp(double a, double b){

        List<Double> vec=new ArrayList<>();
        vec.add(a);
        vec.add(b);

        return Utils.logSumExp(vec);
    }

    public static double[] convertDoubles(List doubles) {
        double[] array = new double[doubles.size()];
        Iterator<Double> iterator = doubles.iterator();
        int i = 0;
        while(iterator.hasNext())
        {
            array[i] = iterator.next();
            i++;
        }

        return array;
    }

    public static List<Double> convertDoubles(double[] doubles) {
        List<Double> list = new ArrayList<Double>();

        for(int i=0; i<doubles.length; i++){

            list.add(doubles[i]);
        }

        return list;
    }

    public static <T extends Number> double add(T x, T y) {
        double sum;
        sum = x.doubleValue() + y.doubleValue();
        return sum;
    }

    public static <T extends Number> double sum(List<T> entries) {

        int N = entries.size();
        double sum=0;

        for(int i=0; i<N; i++)
            sum=add(sum,entries.get(i));

        return sum;
    }

    public static <T extends Number> double mean(List<T> entries) {

        double sum=sum(entries);
        int N = entries.size();

        return sum/N;
    }

    public static List<Double> exp(List entries) {

        int N = entries.size();

        List value= new ArrayList<Double>(N);

        for(int i=0; i<N; i++)
            value.add(Math.exp((double) entries.get(i)));

        return value;
    }

    public static List<Double> log(List entries) {

        int N = entries.size();

        List value= new ArrayList<Double>(N);

        for(int i=0; i<N; i++)
            value.add(Math.log((double) entries.get(i)));

        return value;
    }

    public static List<Double> toListDoubles(List entries) {

        int N = entries.size();

        List value= new ArrayList<Double>(N);

        for(int i=0; i<N; i++)
            value.add((double) entries.get(i));

        return value;
    }

    public static <T extends Number> double variance(List<T> entries) {

        double mean=mean(entries);
        int N = entries.size();

        double var=0;

        for(int i=0; i<N; i++)
            var=add(var,Math.pow(add(entries.get(i),-1*mean),2));

        return var/(N-1);
    }

    public static double range(List<Double> entries) {

        double min=Collections.min(entries);
        double max=Collections.max(entries);

        return max-min;
    }

    public static <T extends Number> List<Double> cumSum(List<T> entries) {

        int N = entries.size();
        List<Double> cumSum=new ArrayList<>(N);
        double sum=0;

        for(int i=0; i<N; i++) {

            sum=add(sum,entries.get(i));
            cumSum.add(sum);
        }

        return cumSum;
    }

    public static List<Double> equalWeights(int N) {

        List<Double> logW=new ArrayList(N);

        for(int i=0; i<N; i++){
            logW.add(i, -Math.log(N));
        }

        return logW;
    }

    public static List<Double> sum(List x, double c) {

        int N=x.size();
        List y=new ArrayList<Double>(N);

        for(int i=0; i<N; i++){
            y.add((double) x.get(i)+c);
        }

        return y;
    }

    public static List<Double> sum(List x, List y) {

        int N=x.size();
        List z=new ArrayList<Double>(N);

        for(int i=0; i<N; i++){
            z.add((double) x.get(i) + (double)y.get(i));
        }

        return z;
    }

    public static List<Double> prod(List x, double c) {

        int N=x.size();
        List y=new ArrayList<Double>(N);

        for(int i=0; i<N; i++){
            y.add((double) x.get(i)*c);
        }

        return y;
    }

    public static void prod(List<Double> y, List x, double c) {

        int N=x.size();

        for(int i=0; i<N; i++){
            y.set(i, (double) x.get(i)*c);
        }
    }

    public static double[][] prod(double[][] X, double c) {

        int nRows=X.length;
        if(nRows==0)
            return X;

        int nCols=X[0].length;

        double[][] mat = new double[nRows][nCols];

        for (int i = 0; i < nRows; i++)
            for(int j=0; j<nCols; j++){
                mat[i][j]=X[i][j]*c;
            }

        return mat;
    }

    public static double[] equalVector(double x, int N) {

        double[] vec = new double[N];

        for (int i = 0; i < N; i++)
            vec[i] = x;

        return vec;
    }

    public static double[][] idMatrix(double s2, int N) {

        double[][] mat = new double[N][N];

        for (int i = 0; i < N; i++)
            for(int j=0; j<N; j++){
                if(i==j)
                    mat[i][j]=s2;
                else
                    mat[i][j]=0;
            }

        return mat;
    }

    public static double[][] idMatrix(List<Double> s2) {

        int N=s2.size();
        double[][] mat = new double[N][N];

        for (int i = 0; i < N; i++)
            for(int j=0; j<N; j++){
                if(i==j)
                    mat[i][j]=s2.get(i);
                else
                    mat[i][j]=0;
            }

        return mat;
    }

    public static int areOrdered(List<Double> x) {

        int N=x.size();

        for(int i=0; i<N-1; i++){

            if(x.get(i)>x.get(i+1))
                return 0;
        }

        return 1;
    }

    public static List<Double> LogitToLog(List<Double> logit_x){

        int N=logit_x.size();
        List log_x=new ArrayList<Double>(N);

        for(int i=0; i<N; i++){
            log_x.add(-1* Math.log(1+Math.exp(-1.0*logit_x.get(i))));
        }

        return log_x;
    }

    public static List<Double> Logit2ToLogit(List<Double> logit2_x){

        int N=logit2_x.size();
        List temp=new ArrayList(logit2_x);
        temp.add(0.0);

        double log_xNp1=-1*logSumExp(temp);

        List log_x=sum(temp.subList(0,N),log_xNp1);

        return LogToLogit(log_x);
    }

    public static double LogitToLog(double logit_x){

        double log_x=-1* Math.log(1+Math.exp(-1.0*logit_x));

        return log_x;
    }

    public static List<Double> LogToLogit(List<Double> log_x){

        int N=log_x.size();
        List logit_x=new ArrayList<Double>(N);

        for(int i=0; i<N; i++){
            logit_x.add(-1*Math.log(Math.exp(-log_x.get(i))-1));
        }

        return logit_x;
    }

    public static double LogToLogit(double log_x){

        double logit_x=-1*Math.log(Math.exp(-log_x)-1);

        return logit_x;
    }

    public static int anyNaN(List<ArrayList<Object>> x){

        int N=x.size();

        for(int i=0; i<N; i++){

            int M=x.get(i).size();

            for(int j=0; j<M; j++){

                if(Double.isNaN((double) x.get(i).get(j)))
                    return i;
            }
        }

        return -1;
    }



    public static int WeightsOutBounds(List<Double> logitW){

        List logW=LogitToLog(logitW);

        if(logW.size()==0)
            return 0;

        if( logSumExp(logW)>0 ){
            return 1;
        }

        return 0;
    }

    public static ArrayList<Double> ValList(List<Double> x){

        ArrayList<Double> val=new ArrayList<>();

        for(double y : x) {

            val.add(y);
        }

        return val;
    }
}
