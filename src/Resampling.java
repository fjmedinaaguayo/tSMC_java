import java.util.*;

interface ResamplingScheme{

    ParticleSet resample(ParticleSet particles);
}

class Systematic implements ResamplingScheme {

    public ParticleSet resample(ParticleSet  particles) {

        int N = particles.logW.size();
        ParticleSet resampledParticles=new ParticleSet(N);

        List<Double> logWnorm=particles.normalisedWeights();

        double u = Math.random();
        double C = 0;
        int j=1;

        for (int i = 0; i < N; i++) {

            C += Math.exp(logWnorm.get(i));
            while((u+j-1.0)/N<=C){

                resampledParticles.th.add(j-1,particles.th.get(i));
                j+=1;
            }
        }

        resampledParticles.logW=Utils.equalWeights(N);

        return resampledParticles;
    }
}