const std = @import("std");
const Random = std.Random;
const pow = std.math.pow;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

const RNGError = error{
    InvalidRange,
    NotAFloat
};


const size = 1000;

pub fn main() !void {
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    var rand = prng.random();
    
    std.debug.print("Let's generate {d} uniform random values\n", .{size});
    
    var uniform_sample: [size]f32 = undefined;
       
    for (0..size) |i| {
        const u = try runif(f32, 0.0, 1.0, &rand);
        uniform_sample[i] = u;
    }

    std.debug.print("Let's generate {d} weibull random values\n", .{size});
     
    var exp_sample: [size]f32 = undefined;
    const lambda: f32 = 2;
    for (0.., uniform_sample) |i, u| {
        exp_sample[i] = lambda*(-@log(u));
    }
    try write_sample_to_file("exp.csv", &exp_sample);

    var weibull_sample: [size]f32 = undefined;
    const lambda: f32 = 2;
    const k: f32 = 10; 
    for (0.., uniform_sample) |i, u| {
        weibull_sample[i] = lambda*(pow(f32, -@log(u), 1.0 / k));
    }
    try write_sample_to_file("weibull.csv", &weibull_sample);
    
    var gamma_sample: [size]f32 = undefined;
    const gamma: f32 = 0.5;
    const x0: f32 = 9;
    for (0.., uniform_sample) |i, u| {
        gamma_sample[i] =  x0 + gamma * @tan(std.math.pi*(u - 0.5));
    }
    try write_sample_to_file("gamma.csv", &gamma_sample);

}

fn write_sample_to_file(name: []const u8, sample: *[size]f32) !void {
    const cwd = std.fs.cwd();
    const file = try cwd.createFile(name, .{});
    defer file.close();

    var buf: [2048]u8 = undefined;
    var stdout_file = file.writer(&buf);
    const writer: *std.Io.Writer = &stdout_file.interface;
    
    for (sample) |s| {
        try writer.print("{d} ", .{s}); 
    } 
    try writer.print("\n", .{});
    try writer.flush();
    return;
}

/// un petit comentari sobre la generació de nombres aleatoris en un ordinador
/// Hi ha diversos algorismes que et permeten generar nombres aleatòriament.
/// zig distingeix entre la interfície std.Random i l'algorisme que fa servir
/// Per tant, la manera de generar nombres aleatoris és primer generar un Random
/// amb la interfície adequada.
///
/// Un amb la interfície Random, hem d'implementar un algorisme per generar
/// el runif, que és treure un nombre aleatori entre min i max 

/// Implementation of random number following the uniform distribution
/// Just accepts f32 or f64 
fn runif(comptime T: type, a: T, b: T, rng: *Random) !T {
    if ((T != f32) and (T != f64)){
        return RNGError.NotAFloat;
    }
    
    if (b < a) {
        return RNGError.InvalidRange;
    }
    // scale if needed 
    return a + (b - a) * rng.float(T);
}

///to mimic R behaviour, we implement an allocator to do it dinamically
fn rwbSampleAlloc(allocator: Allocator, size: n, comptime T: type, lambda: T, k: T, rng: *Random) !ArrayList(T) {
    var sample: ArrayList(T) = .empty;

    for (0..n) |i| {
        const u = try runif(T, 0, 1, rng);
        const w = lambda*(pow(f32, -@log(u), 1.0 / k));
        _ = try sample.append(w);
    }

    return sample;
}

/// the optimal way is not to call runifSampleAlloc and then multiply all the value or the inverse cdf,
/// because the cost of the fetching the data are far worse than the extra two evaluations.
fn runifSampleAlloc(allocator: Allocator, size: n, comptime T: type, a: T, b: T, rng: *Random) !ArrayList(T) {
    var sample: ArrayList(T) = .empty;

    for (0..n) |i| {
        const u = try runif(T, a, b, rng);
        _ = try sample.append(allocator, u);
    }

    return sample;
}


