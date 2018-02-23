import java.io.*;
import java.util.*;
import java.util.stream.IntStream;
import org.apache.commons.cli.*;

public class tSMC_Algo {

    private static int N;
    private static int T,t0=0;
    private static double alpha=0.5, beta=0.999, probBirth=0.5;
    private static Problem problem;
    private static ResamplingScheme resampling=new Systematic();
    private static String filename, inputFile, outputFile, fullOutput;

    public static void main(String[] args) throws ParseException {

        Options options = new Options();
        options.addOption("h", false, "Help");
        options.addOption("N", true, "Number of particles");
        options.addOption("T", true, "Maximum number of components");
        options.addOption("t0", true, "Initial number of components (default=0)");
        options.addOption("a", true, "ESS relative threshold (default=0.5)");
        options.addOption("b", true, "CESS relative threshold (default=0.999)");
        options.addOption("D", true, "Data filename");
        options.addOption("pB", true, "Weight for Birth move (default=00.5)");
        options.addOption("i", true, "Input particles filename");
        options.addOption("o", true, "Output particles filename");
        options.addOption("p", false, "Print all particles");
        options.addOption("w", false, "Without MCMC moves");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse( options, args);

        if(cmd.hasOption("h")){

            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp("Options: ", options);
            System.exit(1);
        }
        else{

            N= Integer.parseInt(cmd.getOptionValue("N"));
            T= Integer.parseInt(cmd.getOptionValue("T"));

            if(cmd.hasOption("t0"))
                t0= Integer.parseInt(cmd.getOptionValue("t0"));
            if(cmd.hasOption("a"))
                alpha= Double.parseDouble(cmd.getOptionValue("a"));
            if(cmd.hasOption("b"))
                beta= Double.parseDouble(cmd.getOptionValue("b"));
            if(cmd.hasOption("pB"))
                probBirth= Double.parseDouble(cmd.getOptionValue("pB"));

            filename= cmd.getOptionValue("D");

            if(cmd.hasOption("i"))
                inputFile= cmd.getOptionValue("i");
            if(cmd.hasOption("o"))
                outputFile= cmd.getOptionValue("o");
        }

        double startTime = System.currentTimeMillis();

        ReadFile rf = new ReadFile();
        ArrayList<Double> Data=new ArrayList<>();

        try
        {
            String[] dataLines = rf.readData(filename);

            for(String line : dataLines)
                Data.add(Double.valueOf(line));
        }
        catch(IOException e)
        {
            // Print out the exception that occurred.
            System.out.println("Unable to read data file, "+e.getMessage());
            System.exit(1);
        }

        double logZ=0.0;
        List<Double> essList=new ArrayList();
        List<Double> relCessList=new ArrayList();
        List<Double> gammaList=new ArrayList();
        List<Integer> intermDist=new ArrayList();
        List<Double> timesResamp=new ArrayList();
        List<Double> logZList=new ArrayList<>();


        WriteFile wf = new WriteFile();
        ParticleSet PSt=new ParticleSet();

//        problem=new SMC2(T,Data);
//        PSt=new ParticleSet(Utils.equalWeights(N),problem.pi0.Draw(N));

        if(cmd.hasOption("i")){

            try{

                List temp=rf.readParticles(inputFile);

                t0=(int) temp.get(0);
                logZ=(double) temp.get(1);
                PSt=(ParticleSet) temp.get(2);
                N=PSt.logW.size();
                fullOutput=outputFile+"_pB"+probBirth+"_N"+N+"_a"+alpha+"_b"+beta;

                problem=new MixGaussian(t0, T, Data, probBirth);

            }catch(IOException e){
                e.printStackTrace();
            }
        }
        else{

            problem=new MixGaussian(t0, T, Data, probBirth);
            PSt=new ParticleSet(Utils.equalWeights(N),problem.pi0.Draw(N));

            if(cmd.hasOption("o") || cmd.hasOption("p")){
                try{

                    fullOutput=outputFile+"_pB"+probBirth+"_N"+N+"_a"+alpha+"_b"+beta;
                    wf.writeToFile(fullOutput+"_t"+problem.t0+".txt",PSt,problem.t0,0,logZ);
                }catch(IOException e){
                    e.printStackTrace();
                }
            }
        }

        for(int t=problem.t0; t<problem.T; t++){

            PSt.setU(problem.psi.get(t).Draw(N));
            PSt = new ParticleSet(PSt.logW,problem.G.get(t).fn(PSt.th));

            if(cmd.hasOption("p")){
                try{

                    wf.writeToFile(fullOutput+"_t"+t+"-fwd"+".txt",PSt,t,0,logZ);
                }catch(IOException e){
                    e.printStackTrace();
                }
            }

            ArrayList<Double> logWinc=new ArrayList(Utils.equalWeights(N));

            double gamma0=0.0;
            double logitGammaInc=-10;
            Integer interm=-1;

            do {

                interm+=1;
                logitGammaInc=Secant(t+1, gamma0,logitGammaInc,logWinc,PSt);
                relCessList.add(PSt.relCESS(logWinc));
                double gamma1=gamma0+(1-gamma0)*Math.exp(Utils.LogitToLog(logitGammaInc));
                gammaList.add(gamma1);

                System.out.print("gamma_{"+t+"To"+(t+1)+"}("+(interm+1)+"): "+gamma1+", ");

                List<Double> tempLogW=PSt.logW;
                IntStream.range(0, N).parallel().
                        forEach( k ->tempLogW.set(k, tempLogW.get(k) + logWinc.get(k)) );

                double ess=PSt.ESS();
                if(ess/N<alpha) {

                    logZ+=Utils.logSumExp(PSt.logW);
                    timesResamp.add((t+1.0+(interm+0.0)/10));
                    PSt=resampling.resample(PSt);
                }
                essList.add(ess);

                int t_dist=t+1;
                TargetDistn phi = (List<Object> x) ->
                        problem.GeomlogPhi_t(gamma1, t_dist, x);

                if(cmd.hasOption("w")==false){

//                    MCMCK MK= new MCMCK(phi,problem.K.get(t));
                    aGRWMetropK MK= new aGRWMetropK(phi,problem.K.get(t));
                    PSt = new ParticleSet(PSt.logW, MK.Draw(PSt.xList()));
                }

                if(cmd.hasOption("p")){
                    try{

                        if(gamma1<1)
                            wf.writeToFile(fullOutput+"_t"+t+"-"+(interm+1)+".txt", PSt,t, gamma1,
                                logZ+Utils.logSumExp(PSt.logW));
                        else
                            wf.writeToFile(fullOutput+"_t"+(t+1)+".txt", PSt,t, gamma1,
                                    logZ+Utils.logSumExp(PSt.logW));

                    }catch(IOException e){
                        e.printStackTrace();
                    }
                }

                gamma0=gamma1;

            } while(gamma0<1.0);

            intermDist.add(interm);
            logZList.add(logZ+Utils.logSumExp(PSt.logW));
            System.out.println("\nlogZ_"+(t+1)+"="+logZList.get(t-problem.t0));
            System.out.println("Estimators:"+PSt.MixGaussEstimates());

            if(cmd.hasOption("o") && cmd.hasOption("p")==false){
                try{

                    wf.writeToFile(fullOutput+"_t"+(t+1)+".txt", PSt,t+1, 0,
                            logZList.get(t-problem.t0));
                }catch(IOException e){
                    e.printStackTrace();
                }
            }
        }

        double endTime = System.currentTimeMillis();

        System.out.println("---\n");
        System.out.println("Elapsed time:" + (endTime - startTime)/60000 + " minutes");
        System.out.println("ESS: "+essList);
        System.out.println("Relative CESS: "+relCessList);
        System.out.println("Resampling times: "+timesResamp);
        System.out.println("Estimate log normalising constant: "+logZList);
        System.out.println("Gamma: "+gammaList);
        System.out.println("Intermediate Distns: "+intermDist);

    }

    public static double Bisection(double gamma0, double gammaInc, List<Double> logWinc, ParticleSet PSt, List<Double> diff_logPhi){

        double logRcess_mod, target=Math.log(Math.abs(Math.log(beta)));
        double gammaL=gamma0, gammaR=gamma0+gammaInc;
        double gammaFinal;

        do {

            double gammaAvg=(gammaL+gammaR)/2;
            gammaFinal=gammaAvg;

            Utils.prod(logWinc, diff_logPhi,gammaAvg-gamma0);
            logRcess_mod=PSt.logRelCESS_mod(logWinc);

            if(logRcess_mod>target)
                gammaR=gammaAvg;
            else
                gammaL=gammaAvg;

        } while(Math.abs(logRcess_mod-target)>1e-6 && (gammaR-gammaL)>1e-9 && gammaL<1.0);

        return Utils.LogToLogit( Math.log(gammaFinal-gamma0)-Math.log(1-gamma0) );
    }

    public static double Secant(int t, double gamma0, double logitGammaInc, List<Double> logWinc_n, ParticleSet PSt){

        double gamma_nm1=-15, gamma_n=logitGammaInc;
        double result, target=Math.log(Math.abs(Math.log(beta)));

        List<Double> diff_logPhi=new ArrayList(Collections.nCopies(N, 0));

        IntStream.range(0, N).parallel().
                forEach( k -> {

            diff_logPhi.set(k,problem.GeomlogPhi_t(1, t, PSt.th.get(k).x)
                    -problem.GeomlogPhi_t(0, t, PSt.th.get(k).x));
        });

        List<Double> logWinc_nm1=Utils.prod(diff_logPhi,(1-gamma0)*Math.exp(Utils.LogitToLog(gamma_nm1)));
        Utils.prod(logWinc_n, diff_logPhi,(1-gamma0)*Math.exp(Utils.LogitToLog(gamma_n)));

        double f_nm1=PSt.logRelCESS_mod(logWinc_nm1);
        double f_n=PSt.logRelCESS_mod(logWinc_n);
        int iters=0;

        do {
            double gamma_np1=gamma_n+(target-f_n)*(gamma_n-gamma_nm1)/(f_n-f_nm1);

            if(Double.isNaN(gamma_np1)){

                System.out.print("Secant method failed -> bisection will be attempted, ");
                gamma_n=Bisection(gamma0,2*(1-gamma0),logWinc_n,PSt,diff_logPhi);

                break;
            }

            gamma_nm1=gamma_n;
            f_nm1=f_n;

            gamma_n=gamma_np1;
            Utils.prod(logWinc_n, diff_logPhi,(1-gamma0)*Math.exp(Utils.LogitToLog(gamma_n)));
            f_n=PSt.logRelCESS_mod(logWinc_n);

            if(gamma_n==Double.POSITIVE_INFINITY){

                if(f_n>target)
                    System.out.println("Secant method failed!");
                break;
            }

            iters+=1;

        } while(Math.abs(f_n-target)>1e-6 && iters<1e+3);

        for(int k=0; k<N; k++){
            if(Double.isNaN( logWinc_n.get(k)) ){

                logWinc_n.set(k,(problem.GeomlogPhi_t(1, t, PSt.th.get(k).x)
                        -problem.GeomlogPhi_t(0, t, PSt.th.get(k).x))*
                        (1-gamma0)*Math.exp(Utils.LogitToLog(gamma_n)) );
            }
        }

        result=gamma_n;
        System.out.print("rCESS: "+PSt.relCESS(logWinc_n)+", ");

        return result;
    }
}