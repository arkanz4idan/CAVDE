use std::io::BufRead;
use serde::Serialize;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::process;
use std::time::Instant;

/// Zero-allocation struct for JSON serialization.
/// Using string slices (`&'a str`) borrows directly from the line buffer,
/// completely avoiding intermediate String allocations during serialization.
#[derive(Serialize)]
struct Record<'a> {
    input: &'a str,
    output: &'a str,
}

// 256KB buffers strike the optimal balance for modern NVMe/SSD drives.
// It drastically reduces syscall frequency while fitting comfortably in CPU L2/L3 cache.
const BUFFER_SIZE: usize = 256 * 1024;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Benchmark Generation Mode: txt2jsonl --generate <lines> <output_file>
    if args.len() == 4 && args[1] == "--generate" {
        let lines: usize = args[2].parse().unwrap_or_else(|_| {
            eprintln!("Error: <lines> must be a valid integer.");
            process::exit(1);
        });
        let output_path = &args[3];

        println!("Generating {} lines to {}...", lines, output_path);
        let start = Instant::now();
        
        let file = File::create(output_path)?;
        let mut writer = BufWriter::with_capacity(BUFFER_SIZE, file);

        for i in 0..lines {
            // Realistic variance: mix of simple text, quotes, and pipes
            if i % 10 == 0 {
                writeln!(writer, "Test input {}|Test \"output\" with \\ backslash", i)?;
            } else if i % 5 == 0 {
                writeln!(writer, "Input {}|Output with | multiple | pipes", i)?;
            } else {
                writeln!(writer, "Simple input {}|Simple output {}", i, i)?;
            }
        }
        writer.flush()?;

        let elapsed = start.elapsed().as_secs_f64();
        let file_size = std::fs::metadata(output_path)?.len() as f64 / (1024.0 * 1024.0);
        println!("Generated {:.2} MB in {:.3} seconds ({:.2} MB/s)", file_size, elapsed, file_size / elapsed);
        return Ok(());
    }

    // Normal Conversion Mode: txt2jsonl <input.txt> <output.jsonl>
    if args.len() != 3 {
        eprintln!(
            "Usage:\n  {} <input.txt> <output.jsonl>\n  {} --generate <lines> <benchmark.txt>",
            args.first().map(|s| s.as_str()).unwrap_or("txt2jsonl"),
            args.first().map(|s| s.as_str()).unwrap_or("txt2jsonl")
        );
        process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let file_in = File::open(input_path)?;
    let file_size = file_in.metadata()?.len();

    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file_in);
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, File::create(output_path)?);

    // Reuse a single String allocation for the entire program lifetime.
    // .clear() resets length to 0 but retains heap capacity, ensuring O(1) memory.
    let mut line = String::new();

    let mut line_number = 0;
    let mut total_lines = 0;
    let mut converted_lines = 0;
    let mut invalid_lines = 0;

    let start = Instant::now();

    loop {
        line.clear(); 
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break; // EOF
        }

        line_number += 1;
        total_lines += 1;

        // O(1) newline stripping without allocation or regex
        let line_content = line.as_str();
        let line_content = line_content.strip_suffix('\n').unwrap_or(line_content);
        let line_content = line_content.strip_suffix('\r').unwrap_or(line_content);

        // split_once stops at the first delimiter, returning zero-allocation slices
        if let Some((input, output)) = line_content.split_once('|') {
            let record = Record { input, output };
            
            // Serialize borrowed slices directly into the buffered writer
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
            
            converted_lines += 1;
        } else {
            eprintln!(
                "Warning: Invalid dataset line at line {}: missing '|' separator. Content: {:?}",
                line_number, line_content
            );
            invalid_lines += 1;
        }
    }

    writer.flush()?;

    let elapsed = start.elapsed().as_secs_f64();
    let seconds = elapsed.max(0.001); // Prevent division by zero

    // Memory metrics: Prove O(1) scaling by reporting max line capacity
    let max_line_memory_kb = (line.capacity() as f64) / 1024.0;
    let fixed_io_overhead_kb = (BUFFER_SIZE * 2) as f64 / 1024.0;

    let mb_processed = (file_size as f64) / (1024.0 * 1024.0);
    let mb_per_sec = mb_processed / seconds;
    let lines_per_sec = (total_lines as f64) / seconds;

    println!("\n=== Conversion Complete ===");
    println!("Total lines:     {}", total_lines);
    println!("Converted lines: {}", converted_lines);
    println!("Invalid lines:   {}", invalid_lines);
    println!("Elapsed time:    {:.3} seconds", elapsed);
    println!("Throughput:      {:.2} lines/sec", lines_per_sec);
    println!("Data processed:  {:.2} MB", mb_processed);
    println!("Read/Write speed:{:.2} MB/sec", mb_per_sec);
    println!("---------------------------");
    println!("Memory Profile (O(1) relative to dataset size):");
    println!("  Max line buffer capacity: {:.2} KB (grows only to longest single line)", max_line_memory_kb);
    println!("  Fixed I/O buffer overhead: {:.2} KB", fixed_io_overhead_kb);
    println!("  Total algorithmic memory:  < {:.2} KB", max_line_memory_kb + fixed_io_overhead_kb);

    Ok(())
}