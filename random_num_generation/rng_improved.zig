const std = @import("std");
const eql = std.mem.eql;
const ArrayList = std.ArrayList;
const Random = std.Random;
const Allocator = std.mem.Allocator;
const pow = std.math.pow;

const RNGError = error{
    InvalidRange,
    NotAFloat
};

const HELP = 
    \\ This program generates four files containing samples of the following distributions:
    \\ - Uniform between 0, 1.
    \\ - Exponential: 
    \\ - Weibull: 
    \\ - Gamma: 
    \\
    \\ The script requires an int n as the number of samples to generate. Feel free to write any number needed.  
;

/// As windows is a shitty platform, we need to allocate memory
/// to make a copy of the command line arguments
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var gpa = arena.allocator();

    // set up the stdout buffer
    var buffer: [256]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&buffer);
    const stdout = &stdout_writer.interface;

    // Parse args into string array (error union needs 'try')
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    if (args.len != 2) {
        try stdout.print("Usage: n <int> sample. Write --help for more\n", .{});
        try stdout.flush();
        std.process.exit(1); 
    }

    if (eql(u8, args[1], "-h") or
        eql(u8, args[1], "--help") or
        eql(u8, args[1], "help"))
    {
        try stdout.print("{s}\n", .{HELP});
        try stdout.flush();
        std.process.exit(1);
    }

    const sample = std.fmt.parseInt(u32, args[1], 10) catch |err| {
        try stdout.print("Invalid number to convert: {any}\n", .{err});
        try stdout.flush();
        std.process.exit(1);
    };

    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    var rand = prng.random();

    const T: type = f64; 
    var parameters_found = true; 
    const maybe_file = std.fs.cwd().openFile("parameters.txt", .{}) catch |err| blk: {
        // Inside the blk we can do any amount of work we want.
        // At the end we must produce a value of type `?std.fs.File`.
        switch (err) {
            error.FileNotFound => {
                try stdout.print(
                    "No parameters.txt file exists, setting predefined values\n",
                    .{},
                );
                parameters_found = false;
            },
            else => {
                // Any other error (permission denied, etc.)
                try stdout.print(
                    "Failed to load parameters.txt ({}) – using defaults\n",
                    .{err},
                );
                parameters_found = false;
            },
        }
        // <<< This is the block’s final expression – it becomes the value
        // returned from the `catch`.  Here we return `null` because we could
        // not obtain a file.
        break :blk null;
    };    

    var a: f64 = undefined;
    var b: f64 = undefined;
    var lambda_w: f64 = undefined;
    var k: f64 = undefined;
    var x0: f64 = undefined;
    var gamma: f64 = undefined;
    var lambda_e: f64 = undefined;


    if (maybe_file) |file| {
        defer file.close();

        var file_buffer: [4096]u8 = undefined;
        var reader = file.reader(&file_buffer);
        var line_no: u8 = 0;
        while (reader.interface.takeDelimiterExclusive('\n')) |line| {
            line_no += 1;
            // Tokenise on space or tab (ignore empty tokens)
            var tok_it = std.mem.splitAny(u8, line, " \t");
            while (tok_it.next()) |tok| {
                if (tok.len == 0) continue;

                if (std.mem.eql(u8, tok, "Uniform")) {
                    // Expect two more tokens: a and b
                    const a_tok = tok_it.next() orelse return error.MalformedUniformLine;
                    const b_tok = tok_it.next() orelse return error.MalformedUniformLine;

                    a = try std.fmt.parseFloat(T, a_tok);
                    b = try std.fmt.parseFloat(T, b_tok);
                    try stdout.print(
                        "Line {d}: Uniform({d:.6}, {d:.6})\n",
                        .{ line_no, a, b },
                    );
                    continue; // go to next line
                }

                if (std.mem.eql(u8, tok, "Weibull")) {
                    const lambda_tok = tok_it.next() orelse return error.MalformedWeibullLine;
                    const k_tok      = tok_it.next() orelse return error.MalformedWeibullLine;

                    lambda_w = try std.fmt.parseFloat(T, lambda_tok);
                    k      = try std.fmt.parseFloat(T, k_tok);
                    try stdout.print(
                        "Line {d}: Weibull(λ={d:.6}, k={d:.6})\n",
                        .{ line_no, lambda_w, k },
                    );
                    continue;
                }

                if (std.mem.eql(u8, tok, "Exponential")) {
                    const lambda_tok = tok_it.next() orelse return error.MalformedExpLine;
                    lambda_e = try std.fmt.parseFloat(T, lambda_tok);
                    try stdout.print(
                        "Line {d}: Exponential(λ={d:.6})\n",
                        .{ line_no, lambda_e },
                    );
                    continue;
                }

                if (std.mem.eql(u8, tok, "Gamma")) {
                    const shape_tok = tok_it.next() orelse return error.MalformedGammaLine;
                    const scale_tok = tok_it.next() orelse return error.MalformedGammaLine;

                    gamma = try std.fmt.parseFloat(T, shape_tok);   // often called “α” (shape)
                    x0     = try std.fmt.parseFloat(T, scale_tok);   // often called “β” (scale)
                    try stdout.print(
                        "Line {d}: Gamma(shape={d:.6}, scale={d:.6})\n",
                        .{ line_no, gamma, x0 },
                    );
                    continue;
                }

                try stdout.print(
                    "Line {d}: warning – unknown token '{s}' – ignoring rest of line.\n",
                    .{ line_no, tok },
                );
                break; // stop processing this line; move to the next one
            }
        } else |err| switch (err) {
            error.EndOfStream => {}, // Normal termination
            else => return err,     // Propagate unexpected errors
        }    
    } else {
        a = 0;
        b = 1;
        lambda_w = 9.0;
        k = 2;
        x0 = 8;
        gamma = 7;
        lambda_e = 2;

    }
    try stdout.flush();


    var unif_sample: ArrayList(T) = try runifSampleAlloc(&gpa, sample, T, a, b, &rand);
    defer unif_sample.deinit(gpa);

    var weibull_sample: ArrayList(T) = try rwbSampleAlloc(&gpa, sample, T, lambda_w, k, &rand);
    defer weibull_sample.deinit(gpa);

    var exp_sample: ArrayList(T) = try rexpSampleAlloc(&gpa, sample, T, lambda_e, &rand);
    defer exp_sample.deinit(gpa);

    var gamma_sample: ArrayList(T) = try rgammaSampleAlloc(&gpa, sample, T, gamma, x0, &rand);
    defer gamma_sample.deinit(gpa);

    try stdout.print("All data generated, writing to file...\n", .{});
    try stdout.flush();
    try write_sample_to_file("uniform.csv", T, unif_sample);
    try write_sample_to_file("weibull.csv", T, weibull_sample);
    try write_sample_to_file("exponential.csv", T, exp_sample);
    try write_sample_to_file("gamma.csv", T, gamma_sample);
}

/// Generate a random number of a uniform distribution
/// in the interval [a,b].
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


// petita nota sobre l'arraylist. A la següent funció, quan es crea una array list crea, en essència, una
// struct amb un slice apuntant al heap i la capacitat d'aquest slice (que és un punter a una array + len).
// Per tant, quan es retorna l'arrayList i no un punter a l'arraylist, el que es fa és una còpia d'aquesta mini
// estructura, ja que al estar a l'stack de la funció no es pot tornar-hi un punter (s'allibrerarà just quan es
// retorni la funció.)
// La còpia d'aquesta estructura fa que necessitis declarar l'array list al main com a var i no com a const, ja 
// que al intentar allibrerar la memòria de const __sample no funcinarà el .deinit() ja que ha de ser mutable.

/// Generate an sample of a uniform distribution [a,b) 
fn runifSampleAlloc(allocator: *Allocator, n: u32, comptime T: type, a: T, b: T, rng: *Random) !ArrayList(T) {
    var sample: ArrayList(T) = .empty;
    try sample.ensureTotalCapacity(allocator.*, n);

    for (0..n) |_| {
        const u = try runif(T, a, b, rng);
        _ = try sample.append(allocator.*, u);
    }

    return sample;
}

/// Generate a sample of a Weibull distribution of lambda and k.
fn rwbSampleAlloc(allocator: *Allocator, n: u32, comptime T: type, lambda: T, k: T, rng: *Random) !ArrayList(T) {
    var sample: ArrayList(T) = .empty;
    try sample.ensureTotalCapacity(allocator.*, n);

    for (0..n) |_| {
        const u = try runif(T, 0, 1, rng);
        const w = lambda*(pow(T, -@log(u), 1.0 / k));
        _ = try sample.append(allocator.*, w);
    }

    return sample;
}

/// Generate a sample of a Exponential distribution of parameter lambda.
fn rexpSampleAlloc(allocator: *Allocator, n: u32, comptime T: type, lambda: T, rng: *Random) !ArrayList(T) {
    var sample: ArrayList(T) = .empty;
    try sample.ensureTotalCapacity(allocator.*, n);

    for (0..n) |_| {
        const u = try runif(T, 0, 1, rng);
        const e = lambda*(-@log(u));
        _ = try sample.append(allocator.*, e);
    }

    return sample;
}

/// Generate a sample of a Gamma distribution of gamma and x0.
fn rgammaSampleAlloc(allocator: *Allocator, n: u32, comptime T: type, gamma: T, x0: T, rng: *Random) !ArrayList(T) {
    var sample: ArrayList(T) = .empty;
    try sample.ensureTotalCapacity(allocator.*, n);

    for (0..n) |_| {
        const u = try runif(T, 0, 1, rng);
        const e = x0 + gamma * @tan(std.math.pi*(u - 0.5));
        _ = try sample.append(allocator.*, e);
    }

    return sample;
}


fn write_sample_to_file(name: []const u8, comptime T: type, sample: ArrayList(T)) !void {
    const cwd = std.fs.cwd();
    const file = try cwd.createFile(name, .{});
    defer file.close();

    var buf: [2048]u8 = undefined;
    var stdout_file = file.writer(&buf);
    const writer: *std.Io.Writer = &stdout_file.interface;
    
    for (sample.items) |s| {
        try writer.print("{d} ", .{s}); 
    } 
    try writer.print("\n", .{});
    try writer.flush();
    return;
}


