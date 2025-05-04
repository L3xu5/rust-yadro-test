/// ВАЖНО: все задания выполнять не обязательно. Что получится то, получится сделать.

/// Задание 1
/// Почему функция example1 зависает?
///
/// Ответ:
/// В рантайме токио только один поток, из-за чего задача a1,
/// которая использует активное ожидание через операцию try_recv в бесконечном цикле, никогда не отпускает поток,
/// а в рантайме токио многозадачность корпоративная, т.е задача a2 никогда не выполнится.
/// Варианты решения:
///     1. сделать больше потоков, чтобы гарантировать параллельное выполнение (очевидный, но плохой вариант,
///         т.к активное ожидание - зло)
///     2. заменить активное ожидание в цикле на конструкцию с await (предпочтительный вариант)
fn example1() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();
    let (sd, mut rc) = tokio::sync::mpsc::unbounded_channel();

    let a1 = async move {
        loop {
            if let Ok(p) = rc.try_recv() {
                println!("{}", p);
                break;
            }
        }
    };
    let h1 = rt.spawn(a1);

    let a2 = async move {
        let _ = sd.send("message");
    };
    let h2 = rt.spawn(a2);
    while !(h1.is_finished() || h2.is_finished()) {}

    println!("execution completed");
}

fn example1_fix1() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    let (sd, mut rc) = tokio::sync::mpsc::unbounded_channel();

    let a1 = async move {
        loop {
            if let Ok(p) = rc.try_recv() {
                println!("{}", p);
                break;
            }
        }
    };
    let h1 = rt.spawn(a1);

    let a2 = async move {
        let _ = sd.send("message");
    };
    let h2 = rt.spawn(a2);
    while !(h1.is_finished() || h2.is_finished()) {}

    println!("execution completed");
}

fn example1_fix2() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();
    let (sd, mut rc) = tokio::sync::mpsc::unbounded_channel();

    let a1 = async move {
        if let Some(p) = rc.recv().await {
            println!("{}", p);
        }
    };
    let h1 = rt.spawn(a1);

    let a2 = async move {
        let _ = sd.send("message");
    };
    let h2 = rt.spawn(a2);
    while !(h1.is_finished() || h2.is_finished()) {}

    println!("execution completed");
}

#[derive(Clone)]
struct Example2Struct {
    value: u64,
    ptr: *const u64,
}

/// Задание 2
/// Какое число тут будет распечатано 32 64 или 128 и почему?
/// 
/// Ответ:
/// Здесь поле ptr использует сырой указатель (не проверяется borrow checker'ом).
/// Мы кладем в это поле (объекта t1) указатель 
/// на t1.value. После клонирования в объект t2 его поле ptr все ещё ссылается на поле value объекта t1.
/// После дропа t1 указатель в t2 все ещё ссылается на ту же память, но теперь там может быть что угодно (UB).
/// Дальнейшее присвоение t2.value = 128 напрямую не влияет на поле ptr, но, чисто технически,
/// это значение может записаться в ту область, где было 64, т.к формально она теперь свободна.
/// Так что скорее всего, выведется 64, но может и 128
/// (маловероятно, т.к структура t2 уже создана в другом месте, но пути аллокатора неисповедимы),
/// а может вообще что угодно, потому что у нас use after free.
fn example2() {

    let num = 32;

    let mut t1 = Example2Struct {
        value: 64,
        ptr: &num,
    };

    t1.ptr = &t1.value;

    let mut t2 = t1.clone();

    drop(t1);

    t2.value = 128;

    unsafe {
        println!("{}", t2.ptr.read());
    }

    println!("execution completed");
}

/// Задание 3
/// Почему время исполнения всех пяти заполнений векторов разное (под linux)?
/// 
/// Ответ:
/// Первые два варианта различаются между собой тем, что во втором аллоцируется сразу весь вектор,
/// что дает выигрыш в скорости. Дело в том, что оценка на вставку в вектор амортизиронная,
/// т.к если не указать начальный размер, вектор создастся с каким-то определенным
/// размером (массив под капотом), а при его исчерпывании создаст уже внутри массив с большим размером
/// и скопирует уже имеющиеся данные туда. Именно из-за этой аллокации и копирования время различается.
/// При этом варианты 1 и 2 очень проигрывают остальным, т.к вместо push используется insert, который
/// вообще за O(n) работает, т.к сдвигает элементы справа (хоть у нас их и нет, но формально сдвиг происходит)
/// В третьем варианте не только сразу выделяется память под весь планируемый вектор,
/// но и сразу указывается, чем его заполнить, а значит компилятор может оптимизировать данный код
/// через более узкие инструкции. А в четвертом варианте не тратится время на выделение памяти
/// (а это очень долго в сравнении с остальным), так ещё мы и в вектор ничего не пишем, т.к
/// просто происходит копирование в локальную переменную elem.
/// В пятом варианте мы сразу выделяем память и говорим, что хотим заполнить её нулями, 
/// а при выделении память уже инициализирована нулями (память внутри раста),
/// поэтому компилятор оптимизирует этот код чисто до выделения памяти, что и дает выигрыш.
/// P.S если собрать под релизом, то видно, что четвертое "заполнение" занимает 0 мс, т.к
/// компилятор увидел, что цикл ничего не делает. А варианты 3 и 5 почти не отличаются,
/// т.к инициализация нулями и значениями почти идентична при должной оптимизации
/// (под капотом все равно раст полученную от системы память все равно инициализирует сам)
fn example3() {
    let capacity = 10000000u64;

    let start_time = std::time::Instant::now();
    let mut my_vec1 = Vec::new();
    for i in 0u64..capacity {
        my_vec1.insert(i as usize, i);
    }
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    let start_time = std::time::Instant::now();
    let mut my_vec2 = Vec::with_capacity(capacity as usize);
    for i in 0u64..capacity {
        my_vec2.insert(i as usize, i);
    }
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    let start_time = std::time::Instant::now();
    let mut my_vec3 = vec![6u64; capacity as usize];
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    let start_time = std::time::Instant::now();
    for mut elem in my_vec3 {
        elem = 7u64;
    }
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    let start_time = std::time::Instant::now();
    let my_vec4 = vec![0u64; capacity as usize];
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    println!("execution completed");
}

/// Задание 4
/// Почему такая разница во времени выполнения example4_async_mutex и example4_std_mutex?
/// 
/// Ответ:
/// Мьютекс из токио асинхронный и мы в цикле каждый раз происходит переключение контекста, т.к
/// мы ждем возможность взять блокировку асинхронно. А мьютекст из стандартной библиотеки ждет
/// блокировку синхронно.
/// При замерах скорости мы смотрим время завершения любой одной первой задачи, запуская на двух потоках.
/// И при использовании мьютекса из std две задачи будут выполняться в двух потоках, а третья начнет
/// выполняться только после завершения одной из двух первых (что уже не будет учитываться в замерах).
/// Т.е мьютекс токио будет равномерно стараться выполнять все три задания, постоянно тратя время
/// на переключение контекста, а мьютекс std будет выполнять два задания в двух потоках,
/// причем из-за маленького размера критической секции блокировка будет освобождаться достаточно быстро,
/// что для данной задачи и данной метрики оценивания намного быстрее.
/// Вообще для этого примера было бы быстрее всего взять блокировку сразу на весь цикл и быстро
/// выполнить одну задачу (дописал пример)
async fn example4_async_mutex(tokio_protected_value: std::sync::Arc<tokio::sync::Mutex<u64>>) {
    for _ in 0..1000000 {
        let mut value = *tokio_protected_value.clone().lock().await;
        value = 4;
    }
}

async fn example4_std_mutex(protected_value: std::sync::Arc<std::sync::Mutex<u64>>) {
    for _ in 0..1000000 {
        let mut value = *protected_value.clone().lock().unwrap();
        value = 4;
    }
}

async fn example4_fastest_mutex(protected_value: std::sync::Arc<std::sync::Mutex<u64>>) {
    let mut value = *protected_value.clone().lock().unwrap();
    for _ in 0..1000000 {
        value = 4;
    }
}

fn example4() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();

    let mut tokio_protected_value = std::sync::Arc::new(tokio::sync::Mutex::new(0u64));

    let start_time = std::time::Instant::now();
    let h1 = rt.spawn(example4_async_mutex(tokio_protected_value.clone()));
    let h2 = rt.spawn(example4_async_mutex(tokio_protected_value.clone()));
    let h3 = rt.spawn(example4_async_mutex(tokio_protected_value.clone()));

    while !(h1.is_finished() || h2.is_finished() || h3.is_finished()) {}
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    let protected_value = std::sync::Arc::new(std::sync::Mutex::new(0u64));

    let start_time = std::time::Instant::now();
    let h1 = rt.spawn(example4_std_mutex(protected_value.clone()));
    let h2 = rt.spawn(example4_std_mutex(protected_value.clone()));
    let h3 = rt.spawn(example4_std_mutex(protected_value.clone()));

    while !(h1.is_finished() || h2.is_finished() || h3.is_finished()) {}
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    let protected_value = std::sync::Arc::new(std::sync::Mutex::new(0u64));

    let start_time = std::time::Instant::now();
    let h1 = rt.spawn(example4_fastest_mutex(protected_value.clone()));
    let h2 = rt.spawn(example4_fastest_mutex(protected_value.clone()));
    let h3 = rt.spawn(example4_fastest_mutex(protected_value.clone()));

    while !(h1.is_finished() || h2.is_finished() || h3.is_finished()) {}
    println!(
        "execution time {}",
        (std::time::Instant::now() - start_time).as_nanos()
    );

    println!("execution completed");
}

/// Задание 5
/// В чем ошибка дизайна? Каких тестов не хватает? Есть ли лишние тесты?
/// 
/// Ответ:
/// Тут есть ошибка в логике взаимодействия. Если мы считаем, что у треугольника не могут меняться стороны,
/// то поля должны быть приватными, а стороны задаваться при инициализации через аргументы new(),
/// тогда оправдана мемоизация вызовов area() и perimeter() через поле.
/// Если мы считаем, что стороны треугольника могут меняться, то может произойти проблема, где мы
/// вычислим площадь, поменяем стороны, потом снова попробуем вычислить площадь, а вернется прошлое значение,
/// поэтому в данном случае лучше пересчитывать площадь каждый раз, либо, если очень надо, каким-то 
/// образом отслеживать изменения.
/// 
/// Теперь посмотрим на тесты. Я буду их анализировать относительно того дизайна, который дан.
/// Тогда странными и ненужными выглядят тесты, где мы создаем новую структуру, а потом
/// присваиваем полям нулевые значения, хотя они и так нулевые после инициализации.
/// Также в тесте площади зачем-то в конце считается площадь и просто выводится, не сравниваясь 
/// с каким-то ожидаемым значением, что выглядит бессмысленно.
/// Как раз не хватает тестов, которые покрывали бы проблемные места реализации:
///     - тест на изменение сторон треугольника и ожидаемое изменение площади/периметра
///         или же наоборот тест на то, что площадь/периметр не меняется
///     - тест на соответствие ожидаемой точности вычислений
/// 
/// Внес описанные выше изменения, исходя из того, что треугольник иммутабельный
mod example5 {
    pub struct Triangle {
        // теперь поля приватные
        a: (f32, f32),
        b: (f32, f32),
        c: (f32, f32),
        area: Option<f32>,
        perimeter: Option<f32>,
    }

    impl Triangle {
        //calculate area which is a positive number
        pub fn area(&mut self) -> f32 {
            if let Some(area) = self.area {
                area
            } else {
                self.area = Some(f32::abs(
                    (1f32 / 2f32) * (self.a.0 - self.c.0) * (self.b.1 - self.c.1)
                        - (self.b.0 - self.c.0) * (self.a.1 - self.c.1),
                ));
                self.area.unwrap()
            }
        }

        fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
            f32::sqrt((b.0 - a.0) * (b.0 - a.0) + (b.1 - a.1) * (b.1 - a.1))
        }

        //calculate perimeter which is a positive number
        pub fn perimeter(&mut self) -> f32 {
            if let Some(perimeter) = self.perimeter {
                perimeter // убрал return
            } else {
                self.perimeter = Some(
                    Triangle::dist(self.a, self.b)
                        + Triangle::dist(self.b, self.c)
                        + Triangle::dist(self.c, self.a),
                );
                self.perimeter.unwrap()
            }
        }

        //new makes no guarantee for a specific values of a,b,c,area,perimeter at initialization, но теперь дает
        pub fn new(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> Triangle { // теперь конструктор принимает стороны как аргументы
            Triangle {
                a,
                b,
                c,
                area: None,
                perimeter: None,
            }
        }
    }
}

#[cfg(test)]
mod example5_tests {
    use super::example5::Triangle;

    #[test]
    fn test_area() {
        let mut t = Triangle::new((0., 0.), (0., 0.), (0., 0.));

        assert_eq!(t.area(), 0f32); // теперь assert_eq! вместо assert!

        let mut t = Triangle::new((0f32, 0f32), (0f32, 1f32), (1f32, 0f32));

        assert_eq!(t.area(), 0.5); // теперь assert_eq! вместо assert!

        let mut t = Triangle::new((0f32, 0f32), (0f32, 1000f32), (1000f32, 0f32));

        assert_eq!(t.area(), 500000.);  // теперь assert_eq! вместо непонятного println!

        for _ in 0..100000 { // проверим, что площадь не меняется (тест глупый, но все же)
            assert_eq!(t.area(), 500000.);
        }
    }

    #[test]
    fn test_perimeter() {
        let mut t = Triangle::new((0., 0.), (0., 0.), (0., 0.));

        assert_eq!(t.perimeter(), 0f32); // теперь assert_eq! вместо assert!

        let mut t = Triangle::new((0f32, 0f32), (0f32, 1f32), (1f32, 0f32));

        assert_eq!(t.perimeter(), 2f32 + f32::sqrt(2f32)); // теперь assert_eq! вместо assert!
        
        for _ in 0..100000 { // проверим, что периметр не меняется (тест глупый, но все же)
            assert_eq!(t.perimeter(), 2f32 + f32::sqrt(2f32));
        }
    }

    #[test]
    fn test_precision() { // тест точности
        let mut t = Triangle::new((0f32, 0f32), (0f32, 1f32), (1f32, 0f32));
        
        let expected_area = 0.5f32;
        let expected_perimeter = 2f32 + f32::sqrt(2f32);
        
        let computed_area = t.area();
        let computed_perimeter = t.perimeter();
        const EPSILON: f32 = 1e-6;
        
        assert!(
            (computed_area - expected_area).abs() < EPSILON,
            "Area mismatch: expected {}, got {}",
            expected_area,
            computed_area
        );

        assert!(
            (computed_perimeter - expected_perimeter).abs() < EPSILON,
            "Perimeter mismatch: expected {}, got {}",
            expected_perimeter,
            computed_perimeter
        );
    }
}

fn main() {
    example1_fix1();
    example1_fix2();
    example4();
}