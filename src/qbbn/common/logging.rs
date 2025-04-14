
#[macro_export]
macro_rules! print_red {
    ($($arg:tt)*) => {
        println!("\x1b[31m{}\x1b[0m", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! print_green {
    ($($arg:tt)*) => {
        println!("\x1b[32m{}\x1b[0m", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! print_yellow {
    ($($arg:tt)*) => {
        println!("\x1b[33m{}\x1b[0m", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! print_blue {
    ($($arg:tt)*) => {
        println!("\x1b[34m{}\x1b[0m", format!($($arg)*));
    };
}