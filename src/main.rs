#![feature(portable_simd)]

use std::ops::{Add, Mul};
use std::path::PathBuf;
use std::simd::{LaneCount, Simd, SimdPartialOrd, SupportedLaneCount};

use clap::{arg, Parser};
use image::{Rgb, RgbImage};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct SimdComplex<const LANE_COUNT: usize>
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    re: Simd<f32, LANE_COUNT>,
    im: Simd<f32, LANE_COUNT>,
}

impl<const LANE_COUNT: usize> SimdComplex<LANE_COUNT>
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    const ZERO: Self = Self {
        re: Simd::from_array([0.0; LANE_COUNT]),
        im: Simd::from_array([0.0; LANE_COUNT]),
    };

    fn new(re: [f32; LANE_COUNT], im: [f32; LANE_COUNT]) -> Self {
        Self {
            re: Simd::from_array(re),
            im: Simd::from_array(im),
        }
    }

    fn abs2(&self) -> Simd<f32, LANE_COUNT> {
        self.re * self.re + self.im * self.im
    }
}

impl<const LANE_COUNT: usize> Add for SimdComplex<LANE_COUNT>
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    type Output = SimdComplex<LANE_COUNT>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<const LANE_COUNT: usize> Mul for SimdComplex<LANE_COUNT>
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// The number of iterations it took to know a number is not in the mandelbrot
/// set. Returns `None` if the number is in the set.
fn unbounded_iters<const LANE_COUNT: usize>(
    c: SimdComplex<LANE_COUNT>,
    max_iters: usize,
) -> [u32; LANE_COUNT]
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    let mut z = SimdComplex::ZERO;
    let mut inside = Simd::<u32, LANE_COUNT>::splat(0);

    let four = Simd::<f32, LANE_COUNT>::splat(4.0);

    for _ in 0..max_iters {
        z = z * z + c;
        let mask = z.abs2().simd_le(four);

        // inside += mask.select(Simd::splat(1), Simd::splat(0));
        inside -= mask.to_int().cast::<u32>();

        if !mask.any() {
            break;
        }
    }

    inside.into()
}

fn iters_to_rgb(iters: usize, max_iters: usize, bands: usize) -> Rgb<u8> {
    let position = if iters == max_iters {
        0.0
    } else {
        (iters as f64 / (max_iters as f64 / bands as f64)).fract()
    };
    color_from_position(position)
}

fn color_from_position(position: f64) -> Rgb<u8> {
    let left_color = Rgb::<u8>::from([40, 40, 40]);
    let right_color = Rgb::<u8>::from([254, 128, 25]);

    let r = (left_color.0[0] as f64
        + position * (right_color.0[0] as f64 - left_color.0[0] as f64))
        as u8;
    let g = (left_color.0[1] as f64
        + position * (right_color.0[1] as f64 - left_color.0[1] as f64))
        as u8;
    let b = (left_color.0[2] as f64
        + position * (right_color.0[2] as f64 - left_color.0[2] as f64))
        as u8;

    Rgb::from([r, g, b])
}

struct Plane<T> {
    lines: Vec<Vec<T>>,
    width: usize,
    height: usize,
}

impl<const LANE_COUNT: usize> Plane<SimdComplex<LANE_COUNT>>
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    fn new(
        x0: f32,
        dx: f32,
        width: usize,
        y0: f32,
        dy: f32,
        height: usize,
    ) -> Self {
        let mut lines = Vec::with_capacity(height);

        // let cdx = LANE_COUNT as f64 * dx;
        let width = width / LANE_COUNT;

        for h in 0..height {
            let mut line = Vec::with_capacity(width);

            let im = y0 + dy * h as f32;

            for w in 0..width {
                let mut res = [0.0; LANE_COUNT];
                res.iter_mut().enumerate().for_each(|(i, re)| {
                    *re = x0 + dx * (w * LANE_COUNT + i) as f32;
                });

                let complex = SimdComplex::new(res, [im; LANE_COUNT]);
                line.push(complex);
            }
            lines.push(line);
        }

        Self {
            lines,
            width,
            height,
        }
    }
}

impl<T> Plane<T> {
    /// Maps a plane of one type to another type.
    fn map<U, F, const N: usize>(self, closure: &F) -> Plane<U>
    where
        F: Fn(T) -> [U; N] + 'static + Send + Sync,
        T: 'static + Send,
        U: 'static + Send,
    {
        let lines = self
            .lines
            .into_par_iter()
            .map(|line| line.into_iter().flat_map(closure).collect())
            .collect();

        Plane {
            lines,
            width: self.width * N,
            height: self.height,
        }
    }
}

impl From<Plane<Rgb<u8>>> for RgbImage {
    fn from(plane: Plane<Rgb<u8>>) -> Self {
        RgbImage::from_fn(plane.width as u32, plane.height as u32, |x, y| {
            plane.lines[y as usize][x as usize]
        })
    }
}

/// Generate a mandelbrot set to the given file.
#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The `x` coordinate of the center of the plane to compute.
    #[arg(long, default_value_t = 0.0)]
    center_x: f32,
    /// The `y` coordinate of the center of the plane to compute.
    #[arg(long, default_value_t = 0.0)]
    center_y: f32,
    /// The width of the plane.
    #[arg(long, default_value_t = 2.0)]
    width: f32,
    /// The width of the plane in pixels.
    #[arg(long, default_value_t = 512)]
    pixel_width: usize,
    /// The height of the plane in pixels.
    #[arg(long, default_value_t = 512)]
    pixel_height: usize,
    /// Maximum number of iterations of the mandelbrot quadratic map.
    #[arg(long, default_value_t = 10000)]
    max_iters: usize,
    /// Number of bands to use for coloring.
    #[arg(long, default_value_t = 16)]
    bands: usize,
    /// The file to write the set to.
    #[arg(long, default_value = "mandel.png")]
    out_file: PathBuf,
}

fn main() {
    let args = Args::parse();

    // generic_main::<1>(&args);
    // generic_main::<2>(&args);
    // generic_main::<4>(&args);
    // generic_main::<8>(&args);
    generic_main::<16>(&args);
    // generic_main::<32>(&args);
}

fn generic_main<const LANE_COUNT: usize>(args: &Args)
where
    LaneCount<LANE_COUNT>: SupportedLaneCount,
{
    let height =
        args.width / (args.pixel_width as f32 / args.pixel_height as f32);

    let x_start = args.center_x - args.width / 2.0;
    let x_end = args.center_x + args.width / 2.0;

    let y_start = args.center_y - height / 2.0;
    let y_end = args.center_y + height / 2.0;

    let dx = (x_end - x_start) / args.pixel_width as f32;
    let dy = (y_end - y_start) / args.pixel_height as f32;

    let plane = Plane::new(
        x_start,
        dx,
        args.pixel_width,
        y_start,
        dy,
        args.pixel_height,
    );

    let args = args.clone();
    let closure = move |c| {
        let iters = unbounded_iters::<LANE_COUNT>(c, args.max_iters);
        iters.map(|i| iters_to_rgb(i as usize, args.max_iters, args.bands))
    };

    let plane = plane.map(&closure);

    // println!("{LANE_COUNT} took {} ms", now.elapsed().as_millis());

    let image = RgbImage::from(plane);
    image.save(args.out_file).expect("should save to file");
}
