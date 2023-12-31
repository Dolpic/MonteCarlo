use rand::prelude::Distribution;
use rand::thread_rng;
use rand::seq::SliceRandom;
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

#[wasm_bindgen]
pub fn run(f_nb:u32, nb_runs:u32, nb_samples:u32, a:f64, b:f64, interval:u32, distribution:&str, alpha:f64, beta:f64, stratification:u32) -> Vec<f64> {
    let mut rng = thread_rng();

    let nb_dims;
    let f: Integrand;

    match f_nb {
        1 => {nb_dims = 1; f = &f_1;},
        2 => {nb_dims = 1; f = &f_2;},
        3 => {nb_dims = 2; f = &f_3;},
        4 => {nb_dims = 3; f = &f_4;},
        5 => {nb_dims = 4; f = &f_5;},
        6 => {nb_dims = 5; f = &f_6;},
        _ => alert_and_panic(format!("Invalid function number : {}", f_nb).as_str())
    }

    match distribution {
        "Uniform" => {
            let dist = distribution::Uniform::new(a, b).unwrap();
            let sampler:Sampler = &mut || (0..nb_dims).map(|_| dist.sample(&mut rng)).collect();
            let pdf:PDF = &|vec:&Vec<f64>| vec.iter().fold(1., |acc, x| acc * dist.pdf(*x));
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        },
        "Beta" => {
            let dist = distribution::Beta::new(alpha, beta).unwrap();
            let sampler:Sampler = &mut || (0..nb_dims).map(|_| dist.sample(&mut rng)*(b-a)+a).collect();
            let pdf:PDF = &|vec:&Vec<f64>| vec.iter().fold(1., |acc, x| acc * dist.pdf( (x-a)/(b-a) )*(1./(b-a)));
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
        "Stratified" => {
            let nb_splits = stratification as f64;
            let dist = distribution::Uniform::new(0., (b-a)/nb_splits).unwrap();

            let mut position_counter = 0;
            let mut positions:Vec<Vec<f64>> = vec![];
            let mut counter: Vec<f64> = vec![0.; nb_dims];

            let mut stop = false;
            while !stop {
                positions.push(counter.clone());
                counter[0] += 1.;
                for i in 0..nb_dims {
                    if counter[i] >= nb_splits {
                        counter[i] = 0.;
                        if i != nb_dims-1 {
                            counter[i+1] += 1.;
                        }else{
                            stop = true;
                        }
                    }
                }
            }
            //log(format!("{:?}", positions).as_str());
            positions.shuffle(&mut rng);

            let sampler:Sampler = &mut || {
                let current_position = positions[position_counter].clone();
                position_counter += 1;
                if position_counter == positions.len(){
                    positions.shuffle(&mut rng);
                    position_counter = 0;
                }
                current_position.iter().map(|x| a + (b-a)/nb_splits*x + dist.sample(&mut rng) ).collect()
            };
            let pdf:PDF = &|vec:&Vec<f64>| {
                vec.iter().fold(1., |acc, x| acc * dist.pdf( (*x-a)%( (b-a)/nb_splits ) )/nb_splits )
            };
            return extract_result(execute_runs(nb_runs, nb_samples, interval, sampler, pdf, &f));
        }
        _ => alert_and_panic(format!("Invalid distribution : {}", distribution).as_str())
    }
}

fn f_1(x:Vec<f64>) -> f64{
    return x[0]*x[0] + 1.;
}

fn f_2(x:Vec<f64>) -> f64{
    return 4.*x[0]*x[0]*x[0] - 3.*x[0]*x[0] + 5.*x[0] + 1.;
}

fn f_3(x:Vec<f64>) -> f64{
    return x[0]*x[0] + x[1]*x[1];
}

fn f_4(x:Vec<f64>) -> f64{
    return x[0]*x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2];
}

fn f_5(x:Vec<f64>) -> f64{
    return 5.*x[0]*x[0]*x[0] - 10.*x[1]*x[1]*x[1] + 3.*x[2]*x[2]*x[2] - 2.*x[3];
}

fn f_6(x:Vec<f64>) -> f64{
    return x[0]*x[0]*x[0] + x[1]*x[1]*x[1] + x[2]*x[2]*x[2] + x[3]*x[3]*x[3] + x[4]*x[4]*x[4];
}
