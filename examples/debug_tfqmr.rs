// Debug TFQMR - Kelley 1995 "Iterative Methods" Algorithm B.4 TFQMR
// This is the template book version (templates for solution of linear systems)
use linger::{sparse::{CooMatrix, CsrMatrix}, DenseVec, LinearOperator};

fn mv(a: &CsrMatrix<f64>, v: &[f64]) -> Vec<f64> {
    let vd = DenseVec::from_vec(v.to_vec());
    let mut out = DenseVec::zeros(v.len());
    a.apply(&vd, &mut out);
    out.as_slice().to_vec()
}
fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| x*y).sum() }
fn norm(a: &[f64]) -> f64 { dot(a,a).sqrt() }
fn res(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = mv(a, x);
    let r: Vec<f64> = ax.iter().zip(b).map(|(ai,bi)| ai-bi).collect();
    norm(&r) / norm(b)
}

fn main() {
    // Templates for the Solution of Linear Systems, Barrett et al. 1994
    // TFQMR pseudocode:
    //
    // r = b - A x0
    // r_tilde = r (or any vec with (r_tilde,r) != 0)
    // rho = (r_tilde, r)
    // u = A*r  (one matvec at init)
    // p = r
    // w = r
    // tau = ||r||
    // theta = 0, eta = 0
    // d = 0
    //
    // for m = 1,2,... :
    //   if m is odd:
    //     sigma = (r_tilde, A*p)  <-- but A*p = u? No...
    //     Actually let's use the variable names from Barrett et al.
    //
    // Actually Barrett template book pseudocode:
    // Compute r = b - A*x0
    // rho_0 = (r~,r0)^2 ... hmm that seems odd
    //
    // Let me just try the two-matvec per step version where:
    // at each outer step we do A*u and A*p where u and p evolve

    // SIMPLE VERSION: Just 2 matvecs per step using CGS structure
    // From Kelley's MATLAB code tfqmr.m available at SIAM:
    //   r = b - A*x; rtilde = r;
    //   tau = norm(r); eta = 0; theta = 0; rho = tau^2;
    //   v = A*r; u = v; d = zeros; w = r; y = r;
    //
    // while not converged:
    //   sigma = rtilde'*v
    //   alpha = rho/sigma
    //   for j = 1:2
    //     w = w - alpha * (j==1 ? u : A*y_half)
    //     ... where y_half = y - alpha*v
    //   BUT this needs y_half defined first
    //
    //   Kelley's actual update (simplified):
    //   y_half = y - alpha * v  [only needed for even half-step]
    //   half-step 1: w -= alpha*u,  d = y + (theta^2*eta/alpha)*d
    //   half-step 2: y = y_half; w -= alpha*A*y; d = y + ...
    //   update: rho_new = rtilde'*w; beta = rho_new/rho; y = w + beta*y;
    //           u = A*y; v = u + beta*(A*y_half + beta*v)
    //
    // WAIT. The key: u = A*y at END, and v = u + beta*(A*y_half + beta*v)
    // This means v_{n+1} = A*y_{n+1} + beta*(A*y_{n+1/2} + beta*v_n)
    // That's different from just A*y_{n+1}!

    let n = 5usize;
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0);
        if i > 0     { coo.push(i, i-1, -1.0); }
        if i+1 < n   { coo.push(i, i+1, -1.0); }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b = vec![1.0f64; n];
    let mut x = vec![0.0f64; n];

    let r0: Vec<f64> = b.clone(); // x0 = 0 so r0 = b

    let r_tilde = r0.clone();
    let mut w  = r0.clone();
    let mut y  = r0.clone();
    // u = A*y at init
    let ay0 = mv(&a, &y);
    let mut u  = ay0.clone();
    let mut v  = ay0.clone();  // v = A*r0 initially
    let mut d  = vec![0.0f64; n];

    let mut tau   = norm(&r0);
    let mut theta = 0.0f64;
    let mut eta   = 0.0f64;
    // rho = (r_tilde, r0)
    let mut rho   = dot(&r_tilde, &r0);

    println!("Initial: rel_res={:.4e}", res(&a, &x, &b));

    for step in 0..30 {
        let sigma = dot(&r_tilde, &v);
        if sigma.abs() < 1e-14 { break; }
        let alpha = rho / sigma;

        // y_half = y - alpha * v
        let y_half: Vec<f64> = (0..n).map(|l| y[l] - alpha * v[l]).collect();

        // A*y_half
        let ay_half = mv(&a, &y_half);

        // Half-step 1 (odd m = 2k+1): uses u = A*y_k (from previous end or init)
        for l in 0..n { w[l] = w[l] - alpha * u[l]; }
        let coeff1 = theta*theta*eta/alpha;
        for l in 0..n { d[l] = y[l] + coeff1 * d[l]; }
        theta = norm(&w) / tau;
        let c1 = 1.0 / (1.0 + theta*theta).sqrt();
        tau = tau * theta * c1;
        eta = c1*c1 * alpha;
        for l in 0..n { x[l] = x[l] + eta * d[l]; }
        let rel1 = res(&a, &x, &b);
        println!("  m={} tau={:.3e} EXACT_rel={:.4e}", 2*step+1, tau, rel1);
        if rel1 < 1e-10 { println!("CONVERGED"); return; }

        // Half-step 2 (even m = 2k+2): uses y_half and A*y_half
        for l in 0..n { w[l] = w[l] - alpha * ay_half[l]; }
        let coeff2 = theta*theta*eta/alpha;
        for l in 0..n { d[l] = y_half[l] + coeff2 * d[l]; }
        theta = norm(&w) / tau;
        let c2 = 1.0 / (1.0 + theta*theta).sqrt();
        tau = tau * theta * c2;
        eta = c2*c2 * alpha;
        for l in 0..n { x[l] = x[l] + eta * d[l]; }
        let rel2 = res(&a, &x, &b);
        println!("  m={} tau={:.3e} EXACT_rel={:.4e}", 2*step+2, tau, rel2);
        if rel2 < 1e-10 { println!("CONVERGED"); return; }

        // Update rho, y, u, v
        let rho_new = dot(&r_tilde, &w);
        let beta = rho_new / rho;
        rho = rho_new;

        // y = w + beta * y_half
        for l in 0..n { y[l] = w[l] + beta * y_half[l]; }
        // u = A*y (1 matvec)
        u = mv(&a, &y);
        // v = u + beta*(ay_half + beta*v)  [CGS recurrence, 2 matvecs total: u and ay_half above]
        for l in 0..n { v[l] = u[l] + beta * (ay_half[l] + beta * v[l]); }
    }
    println!("Final rel_res = {:.4e}", res(&a, &x, &b));
}
