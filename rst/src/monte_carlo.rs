pub mod monte_carlo {

    pub type Sampler<'a> = &'a mut dyn FnMut()->f64;
    pub type PDF<'a> = &'a dyn Fn(f64)->f64;

    pub fn iterate(nb_samples:u32, interval:u32, sampler:Sampler, pdf:PDF) -> Result<Vec<f64>, String>{
        let mut result:Vec<f64> = vec![]; 
        let mut sum:f64 = 0.;

        for i in 0..nb_samples {
            let x = sampler();

            let pdf = pdf(x);
            if pdf == 0. {
                return Err(format!("The PDf returned 0 for the sample value {}", x))
            }

            sum += f(x)/pdf;

            if i % interval == 0 && i != 0 {
                 result.push(evaluate(sum, i as f64));
            } 
        }
        result.push(evaluate(sum, nb_samples as f64));
        return Ok(result);
    }

    // x^2 + 2*x + 1
    fn f(val:f64) -> f64{
        return 4.*val*val*val - 3.*val*val + 5.*val + 1.;
    }

    fn evaluate(evaluator:f64, nb_samples:f64) -> f64{
        return evaluator / nb_samples;
    }
}
