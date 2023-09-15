use rand::prelude::Distribution;
use rand::thread_rng;
use statrs::distribution::{self, Continuous};
use wasm_bindgen::prelude::*;

mod monte_carlo;
use monte_carlo::monte_carlo::{Sampler, PDF, iterate};


#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn run(nb_runs:u32, nb_samples:u32, a:f64, b:f64, interval:u32, distribution:&str, alpha:f64, beta:f64) -> Vec<f64> {
    let mut rng = thread_rng();

    match distribution {
        "Uniform" => {
            let dist = distribution::Uniform::new(a, b).unwrap();
            let mut sampler = || dist.sample(&mut rng);
            let pdf = |x|dist.pdf(x);
            return execute_runs(nb_runs, nb_samples, interval, &mut sampler, &pdf);
        },
        "Beta" => {
            let dist = distribution::Beta::new(alpha, beta).unwrap();
            let mut sampler = || dist.sample(&mut rng)*(b-a)+a;
            let pdf = |x:f64| dist.pdf( (x-a)/(b-a) )*(1./(b-a));
            return execute_runs(nb_runs, nb_samples, interval, &mut sampler, &pdf);
        },
        "Linear" => {
            let mut counter:f64 = 0.;

            let mut sampler = || {
                let res=counter;
                counter += (b-a)/nb_samples as f64;
                if counter+a >= b {
                    counter = (b-a)/nb_samples as f64;
                    a
                }else{
                    res+a
                }
            };
            let pdf = |_x:f64| 1./(b-a);
            return execute_runs(nb_runs, nb_samples, interval, &mut sampler, &pdf);
        }
        _ => alert_and_panic(format!("Invalid distribution : {}", distribution).as_str())
    }
}

fn execute_runs(nb_runs:u32, nb_samples:u32, interval:u32, sampler:Sampler, pdf:PDF) -> Vec<f64>{
    
    let nb_points: usize = (nb_samples as f32/interval as f32).floor() as usize;
    let nb_runs_f64 = nb_runs as f64;
    let mut sum: Vec<f64> = vec![0.; nb_points];
    let mut sum_squared: Vec<f64> = vec![0.; nb_points];
    let mut first: Vec<f64> = vec![0.; nb_points];
    let mut average: Vec<f64> = vec![0.; nb_points];
    let mut variance: Vec<f64> = vec![0.; nb_points];
        
    for i in 0..nb_runs {
        
        let current = extract_result(iterate(nb_samples, interval, sampler, &pdf));
        
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
    return first;
}

fn extract_result(res:Result<Vec<f64>, String>) -> Vec<f64>{
    match res {
        Ok(x) => return x,
        Err(msg) => alert_and_panic(msg.as_str())
    }
}

fn alert_and_panic(msg: &str) -> !{
    alert(msg);
    panic!();
}
