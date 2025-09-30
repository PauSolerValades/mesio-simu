const std = @import("std");
const Random = std.Random;
const pow = std.math.pow;

const RNGError = error{
    InvalidRange,
    NotAFloat
};

pub fn main() !void {
    // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    // defer arena.deinit();
    // const allocator = arena.allocator();

    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    var rand = prng.random();
    
    const size = 1000;
    std.debug.print("Let's generate {d} uniform random values\n", .{size});
    
    var uniform_sample: [size]f32 = undefined;
    
    for (0..size) |i| {
        const u = try runif(f32, 0.0, 1.0, &rand);
        uniform_sample[i] = u;
    }

    //std.debug.print("{any}\n", .{uniform_sample});
    

    //random_values <- lambda*(-log(U))^(1/k)

    std.debug.print("Let's generate {d} weibull random values\n", .{size});
    
    var weibull_sample: [size]f32 = undefined;
    const lambda = 2;
    const k: f32 = 10;
    for (0.., uniform_sample) |i, u| {
        weibull_sample[i] = lambda*(pow(f32, -@log(u), 1.0 / k));
        std.debug.print("{d} {d}\n", .{u, weibull_sample[i]});
    }

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
        std.debug.print("a must be lower than b\n", .{});
        return RNGError.InvalidRange;
    }
    
    return a + (b - a) * rng.float(T);
}
