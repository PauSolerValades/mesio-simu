const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Random = std.Random;

const heap = @import("structheap.zig");
const sampling = @import("rng.zig");

const EventType = enum { arrival, service, boarding };

const Event = struct {
    time: f64,
    type: EventType,
    id: u64,
};

/// Okay, aquí estaria la màgia...
/// TODO EXPLICAR BÉ
/// En essència la unió només conté una de les tres quan s'inicialitza
pub const Distribution = union(enum) {
    constant: f64,
    exponential: f64,
    uniform: struct { min: f64, max: f64 },

    pub fn sample(self: Distribution, rng: Random) !f64 {
        switch (self) {
            .constant => |val| return val,
            .exponential => |lambda| return sampling.rexp(f64, lambda, rng),
            .uniform => |p| return try sampling.runif(f64, p.min, p.max, rng),
        }
    }

    // Helper to get integer capacity (e.g. 3.0 -> 3)
    pub fn sampleInt(self: Distribution, rng: Random) !u64 {
        const samp = try self.sample(rng);
        return @as(u64, @intFromFloat(@round(samp)));
    }
};

/// Hi havia com moltes variables. He decidit per el "diccionari"
/// que diu l'Esteve però molt més eficient, és a dir, amb memòria
/// contigua
pub const SimConfig = struct {
    passenger_interarrival: Distribution, // distribució que segueix l'arrivada de passatjers
    bus_interarrival: Distribution, // distribució que segueix l'arrivada d'autobusos
    bus_capacity: Distribution, // distribució que segueix la capacitat de l'autobus
    boarding_time: Distribution, // distribució que segueix el temps de pujada d'un passatjer al bus
    system_capacity: u64, // sempre serà un nombre
    horizon: f64, // temps que dura la simulació
};

pub const SimResults = struct {
    duration: f64,
    mean_queue_length: f64,
    mean_system_length: f64,
    mean_service_time: f64,
    lost_passengers: u64,
    processed_events: u64,
    users_report: ArrayList(User), //recordaque l'array list és un fat pointer, quan retornes això només estas copiant un punter a items i capaciy
    traca: ArrayList(Event),
};

const User = struct {
    arrival: f64,
    leaving: ?f64 = null,
    wait_time: ?f64 = null,
    service_time: ?f64 = null,
};

pub fn eventSchedulingBus(gpa: Allocator, random: Random, config: SimConfig) !SimResults {
    var hp = heap.Heap(Event).init();
    defer hp.deinit(gpa);

    var processed_events: u64 = 0;
    var t_clock: f64 = 0.0;

    // variables d'estat globals
    var num_passengers_queue: u64 = 0;
    var current_bus_capacity: u64 = 0;
    var lost_passengers: u64 = 0;
    var realized_bus_capacity: u64 = 0.0;
    var total_service_time: f64 = 0.0;

    var area_queue: f64 = 0.0;
    var area_system: f64 = 0.0;
    var last_event_time: f64 = 0.0;
    var event_id_counter: u64 = 0;
    var served_users_count: u64 = 0;

    // primera arribada de passatjer per començar la simulació
    const t_p = try config.passenger_interarrival.sample(random);
    event_id_counter += 1;
    try hp.push(gpa, Event{ .time = t_p, .type = .arrival, .id = event_id_counter });

    // primera arribada de bus per començar la simulació
    const t_b = try config.bus_interarrival.sample(random);
    event_id_counter += 1;
    try hp.push(gpa, Event{ .time = t_b, .type = .service, .id = event_id_counter });

    // guardem els passatjers amb quan arriben a la parada, quan marxen i la diferència
    // guardem l'ordre de tots els esdeveniments que han passat
    var traca: ArrayList(Event) = .empty;

    var passangers_in_queue: ArrayList(User) = .empty; // no es crida deinit perque la retornem a la strcut
    var first_user_in_queue: usize = 0;

    while (t_clock <= config.horizon and hp.len() > 0) : (processed_events += 1) {
        const next_event = hp.pop().?; // we use ? because we are pretty sure that cannot fail
        t_clock = next_event.time;
        try traca.append(gpa, next_event);

        const dt = t_clock - last_event_time;

        area_queue += @as(f64, @floatFromInt(num_passengers_queue)) * dt;

        const people_on_bus = realized_bus_capacity - current_bus_capacity;
        const system_size = num_passengers_queue + people_on_bus;
        area_system += @as(f64, @floatFromInt(system_size)) * dt;

        last_event_time = t_clock;

        switch (next_event.type) {
            EventType.arrival => { // passanger arrives

                event_id_counter += 1;
                const time_passanger = try config.passenger_interarrival.sample(random);
                const next_time = t_clock + time_passanger;

                try hp.push(gpa, Event{
                    .time = next_time,
                    .type = .arrival, //hostia que guapo
                    .id = event_id_counter,
                });

                // if the sistem is full, client is lost
                if (num_passengers_queue >= config.system_capacity) {
                    lost_passengers += 1;
                } else {
                    num_passengers_queue += 1; //len de passangers_in_queue
                    try passangers_in_queue.append(gpa, User{ .arrival = t_clock });
                }
            },
            EventType.service => { // bus arrives
                realized_bus_capacity = try config.bus_capacity.sampleInt(random);
                current_bus_capacity = realized_bus_capacity;

                event_id_counter += 1;
                const time_bus = try config.bus_interarrival.sample(random);
                const next_bus_time = t_clock + time_bus;

                try hp.push(gpa, Event{ .time = next_bus_time, .type = .service, .id = event_id_counter });

                // Aquí hi ha un fix a preguntar:
                // Si podem començar el boarding directament ho fem oi?
                // (s'ha afegit el current_bus_capacity) en comptes de l'altre
                if (num_passengers_queue > 0 and current_bus_capacity > 0) {
                    event_id_counter += 1;
                    const duration = try config.boarding_time.sample(random);

                    try hp.push(gpa, Event{ .time = t_clock + duration, .type = .boarding, .id = event_id_counter });
                }
            },
            EventType.boarding => { // passatjer ha pujat a l'autobus
                if (num_passengers_queue > 0 and current_bus_capacity > 0) {
                    // update leaving time of the queue
                    const leaving_user: *User = &passangers_in_queue.items[first_user_in_queue];
                    leaving_user.leaving = t_clock;
                    leaving_user.wait_time = t_clock - leaving_user.arrival;
                    first_user_in_queue += 1;

                    num_passengers_queue -= 1;
                    current_bus_capacity -= 1;

                    // si encara hi ha passatjers a la marquesina i el bus no és ple (segon fix a preguntar)
                    if (num_passengers_queue > 0 and current_bus_capacity > 0) {
                        event_id_counter += 1;
                        const duration = try config.boarding_time.sample(random);

                        try hp.push(gpa, Event{ .time = t_clock + duration, .type = .boarding, .id = event_id_counter });
                    } else { // Si la capacitat del bus és 0, aleshores marxa
                        const passengers_on_bus = realized_bus_capacity - current_bus_capacity;

                        const start_index: usize = first_user_in_queue - passengers_on_bus;
                        for (start_index..first_user_in_queue) |i| {
                            var processed_user: *User = &passangers_in_queue.items[i];
                            const s_time = t_clock - processed_user.leaving.?;
                            processed_user.service_time = s_time;

                            total_service_time += s_time;
                            served_users_count += 1;
                        }
                        realized_bus_capacity = 0;
                        current_bus_capacity = 0;
                    }
                }
            },
        }
    }

    const mean_service = if (served_users_count > 0) total_service_time / @as(f64, @floatFromInt(served_users_count)) else 0.0;

    return SimResults{
        .mean_queue_length = area_queue / t_clock,
        .mean_system_length = area_system / t_clock,
        .mean_service_time = mean_service, // Ws
        .duration = t_clock,
        .lost_passengers = lost_passengers,
        .processed_events = processed_events,
        .users_report = passangers_in_queue,
        .traca = traca,
    };
}

pub fn main() !void {
    var gpa_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_allocator.deinit();
    const gpa = gpa_allocator.allocator();

    // set up the stdout buffer
    var buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&buffer);
    const stdout = &stdout_writer.interface;

    var prng = Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rng = prng.random();

    // per a una cua M/M^[X]/1/K considerem la configuració següent
    // Arribades de passatgers: exp(lambda)
    // Serveis de busos: exp(mu)
    // Capacitat del bus: X
    // Maxim nombre de persones a la marquesina: K

    const config = SimConfig{
        .horizon = 100000.0,
        .passenger_interarrival = Distribution{ .exponential = 5.0 }, // lambda
        .bus_interarrival = Distribution{ .exponential = 4.0 }, // mu
        .bus_capacity = Distribution{ .constant = 3.0 }, // X
        .boarding_time = Distribution{ .constant = 1e-16 }, // minim perque no importa
        .system_capacity = 9, // K
    };

    try stdout.print("SIMULATION START\n", .{});
    try stdout.flush();

    var results = try eventSchedulingBus(gpa, rng, config);
    defer results.users_report.deinit(gpa);
    defer results.traca.deinit(gpa);
    var acc_wait: f64 = 0;
    var counter_wait: usize = 0;
    for (results.users_report.items) |user| {
        if (user.wait_time) |wait_time| {
            acc_wait += wait_time;
            counter_wait += 1;
        }
    }
    const mean_wait_time = if (counter_wait > 0) acc_wait / @as(f64, @floatFromInt(counter_wait)) else 0.0;

    try stdout.print("\tDuration: \t\t{d:.4} \n", .{results.duration});
    try stdout.print("\tEvents processed: \t{d} \n", .{results.processed_events});
    try stdout.print("\tLost passengers: \t{d}\n", .{results.lost_passengers});
    try stdout.print("\tMean Queue Length (Lq):   {d:.4}\n", .{results.mean_queue_length});
    try stdout.print("\tMean System Length (L):   {d:.4}\n", .{results.mean_system_length});
    try stdout.print("\tMean Wait Time (Wq):      {d:.4}\n", .{mean_wait_time});
    try stdout.print("\tMean Service Time (Ws):   {d:.4}\n", .{results.mean_service_time});

    try stdout.flush();
}
