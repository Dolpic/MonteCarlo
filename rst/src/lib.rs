use rand::prelude::Distribution;
use rand::thread_rng;
use statrs::distribution::{self, Continuous};
use wasm_bindgen::prelude::*;

mod monte_carlo;
use monte_carlo::monte_carlo::{Sampler, PDF, Integrand, execute_runs};

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn run(f_nb:u32, nb_runs:u32, nb_samples:u32, a:f64, b:f64, interval:u32, distribution:&str, alpha:f64, beta:f64) -> Vec<f64> {
    let mut rng = thread_rng();

    let nb_dims;
    let f: Integrand;

    match f_nb {
        1 => {
            nb_dims = 1;
            f = &f_1;
        },
        2 => {
            nb_dims = 1;
            f = &f_2;
        },
        3 => {
            nb_dims = 2;
            f = &f_3;
        },
        4 => {
            nb_dims = 3;
            f = &f_4;
        },
        _ => alert_and_panic(format!("Invalid function number : {}", f_nb).as_str())
    }

    match distribution {
        "Uniform" => {
            let dist = distribution::Uniform::new(a, b).unwrap();
            let sampler:Sampler = &mut || {
                let mut res:Vec<f64> = vec![];
                for _ in 0..nb_dims {
                    res.push(dist.sample(&mut rng));
                }
                return res;
            };
            let pdf:PDF = &|vec:&Vec<f64>| {
                let mut res = 1.;
                for x in vec.iter(){
                    res *= dist.pdf(*x);
                }
                return res;
            };
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        },
        "Beta" => {
            let dist = distribution::Beta::new(alpha, beta).unwrap();
            let sampler:Sampler = &mut || {
                let mut res:Vec<f64> = vec![];
                for _ in 0..nb_dims {
                    res.push(dist.sample(&mut rng)*(b-a)+a);
                }
                return res;
            };
            let pdf:PDF = &|vec:&Vec<f64>| {
                let mut res = 1.;
                for x in vec.iter() {
                    res *= dist.pdf( (x-a)/(b-a) )*(1./(b-a));
                }
                return res;
            };
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        },
        "Linear" => {
            let mut counter:Vec<f64> = vec![a; nb_dims];
            let increment = (b-a) / (nb_samples as f64).powf(1./nb_dims as f64);

            let sampler:Sampler = &mut || {
                let res = counter.clone();
                counter[0] += increment;
                for i in 0..nb_dims {
                    if counter[i] >= b {
                        counter[i] = a;
                        if i != nb_dims-1 {
                            counter[i+1] += increment;
                        }
                    }
                }
                res
            };
            let pdf:PDF = &|_x:&Vec<f64>| (1./(b-a)).powi(nb_dims as i32);
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        }
        _ => alert_and_panic(format!("Invalid distribution : {}", distribution).as_str())
    }
}

fn f_1(x:Vec<f64>) -> f64{
    return x[0]*x[0] + 1.;
}

fn f_2(x:Vec<f64>) -> f64{
    let x = x[0];
    return 4.*x*x*x - 3.*x*x + 5.*x + 1.;
}

fn f_3(x:Vec<f64>) -> f64{
    return x[0]*x[0] + x[1]*x[1];
}

fn f_4(x:Vec<f64>) -> f64{
    return x[0]*x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2];
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
