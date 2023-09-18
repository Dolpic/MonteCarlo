use rand::prelude::Distribution;
use rand::thread_rng;
use statrs::distribution::{self, Continuous};
use wasm_bindgen::prelude::*;

mod monte_carlo;
use monte_carlo::monte_carlo::{Sampler, PDF, execute_runs};

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
            let sampler:Sampler = &mut || vec![dist.sample(&mut rng)];
            let pdf:PDF = &|x:&Vec<f64>| dist.pdf(x[0]);
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        },
        "Beta" => {
            let dist = distribution::Beta::new(alpha, beta).unwrap();
            let sampler:Sampler = &mut || vec![dist.sample(&mut rng)*(b-a)+a];
            let pdf:PDF = &|x:&Vec<f64>| dist.pdf( (x[0]-a)/(b-a) )*(1./(b-a));
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        },
        "Linear" => {
            let mut counter:f64 = 0.;

            let sampler:Sampler = &mut || {
                let res=counter;
                counter += (b-a)/nb_samples as f64;
                if counter+a >= b {
                    counter = (b-a)/nb_samples as f64;
                    vec![a]
                }else{
                    vec![res+a]
                }
            };
            let pdf:PDF = &|_x:&Vec<f64>| 1./(b-a);
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        }
        _ => alert_and_panic(format!("Invalid distribution : {}", distribution).as_str())
    }
}

// x^2 + 2*x + 1
fn f(x:Vec<f64>) -> f64{
    let x = x[0];
    return 4.*x*x*x - 3.*x*x + 5.*x + 1.;
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
