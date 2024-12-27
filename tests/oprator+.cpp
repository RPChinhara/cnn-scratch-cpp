

int main() {
    tensor a = fill({2, 2}, 1);
    tensor b = fill({1, 2, 2}, 1);
    tensor c = fill({2, 1}, 1);

    std::cout << a << "\n";
    std::cout << b << "\n";

    std::cout << a + b << "\n";
    std::cout << b + a << "\n";

    std::cout << a + c << "\n";
    std::cout << c + a << "\n";

    std::cout << c + b << "\n";

    return 0;
}