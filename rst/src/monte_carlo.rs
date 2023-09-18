pub mod monte_carlo {

    pub type Sampler<'a> = &'a mut dyn FnMut()->Vec<f64>;
    pub type PDF<'a> = &'a dyn Fn(&Vec<f64>)->f64;
    pub type Integrand<'a> = &'a dyn Fn(Vec<f64>)->f64;

    fn iterate(nb_samples:u32, interval:u32, sampler:Sampler, pdf:PDF, f:Integrand) -> Result<Vec<f64>, String>{
        let mut result:Vec<f64> = vec![]; 
        let mut sum:f64 = 0.;

        for i in 0..nb_samples {
            let x = sampler();

            let pdf = pdf(&x);
            if pdf == 0. {
                return Err(format!("The PDf returned 0 for the sample value {:?}", x));
            }

            sum += f(x)/pdf;

            if i % interval == 0 && i != 0 {
                 result.push(evaluate(sum, i as f64));
            } 
        }
        result.push(evaluate(sum, nb_samples as f64));
        return Ok(result);
    }

    fn evaluate(evaluator:f64, nb_samples:f64) -> f64{
        return evaluator / nb_samples;
    }

    pub fn execute_runs(nb_runs:u32, nb_samples:u32, interval:u32, sampler:Sampler, pdf:PDF, f:Integrand) -> Result<Vec<f64>, String>{
        let nb_points: usize = (nb_samples as f32/interval as f32).floor() as usize;
        let nb_runs_f64 = nb_runs as f64;
        let mut sum: Vec<f64> = vec![0.; nb_points];
        let mut sum_squared: Vec<f64> = vec![0.; nb_points];
        let mut first: Vec<f64> = vec![0.; nb_points];
        let mut average: Vec<f64> = vec![0.; nb_points];
        let mut variance: Vec<f64> = vec![0.; nb_points];
            
        for i in 0..nb_runs {
            
            let current = match iterate(nb_samples, interval, sampler, &pdf, &f) {
                Ok(val) => val,
                Err(str) => return Err(str)
            };

            if i == 0 {
                first = current.clone();
            } 
            for i in 0..nb_points {
                sum[i] += current[i];
                sum_squared[i] += current[i] * current[i];
            }
        }
        for i in 0..nb_points {
            average[i] = sum[i] / nb_runs_f64;
            variance[i] = sum_squared[i] / nb_runs_f64 - (sum[i]*sum[i]) / (nb_runs_f64*nb_runs_f64);
        }
        first.append(&mut average);
        first.append(&mut variance);
        return Ok(first);
    }
}
