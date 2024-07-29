import re  # Thư viện này dùng để lọc số thuộc dạng str
import vietnam_number

def changenumvi(textnumchange):
    units = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    unitsmixfortens = ["", "mốt", "hai", "ba", "bốn", "lăm", "sáu", "bảy", "tám", "chín"]  # list này chỉnh sửa lại mốt và lăm khi đọc số chục vd bốn mươi mốt, chín mươi lăm
    tens = ["", "mười", "hai mươi", "ba mươi", "bốn mươi", "năm mươi", "sáu mươi", "bảy mươi", "tám mươi", "chín mươi"]
    hundreds = ["không trăm", "một trăm", "hai trăm", "ba trăm", "bốn trăm", "năm trăm", "sáu trăm", "bảy trăm", "tám trăm", "chín trăm"]

    # Mở rộng danh sách large_units
    large_units = ["", "nghìn", "triệu", "tỉ", "nghìn tỉ", "triệu tỉ", "tỉ tỉ", "nghìn tỉ tỉ", "triệu tỉ tỉ", "tỉ tỉ tỉ"]

    def two_digit_number(n):  # 10a # Thực hiện trả về từ một đến chín mươi chín.
        if n < 10:  # Thực hiện trả về một,hai,... nếu n nhỏ hơn 10
            return units[n]
        elif n < 20:  # Thực hiện trả về mười một, mười hai,... nếu n nhỏ hơn 20
            return "mười " + units[n - 10]
        else:  # Thực hiện trả về hai mươi, hai mốt,...,chín mươi chín nếu n nhỏ hơn 99
            ten = n // 10  # (// là chia lấy phần nguyên) đặt giá trị của ten = số chục ví dụ 32//10 là sẽ bằng 3
            unit = n % 10  # (% là chia lấy dư) đặt giá trị của unit = số đơn vị ví dụ 32%10 là sẽ bằng 2 vì dư 2
            if unit != 0:
                return tens[ten] + " " + unitsmixfortens[unit]
            else:
                return tens[ten]

    def three_digit_number(n):  # 10b # Thực hiện trả về từ một trăm đến chín trăm mươi chín.
        hundred = n // 100
        rest = n % 100
        if rest == 0:
            return hundreds[hundred]
        elif rest < 10:  # Thêm điều kiện này để đọc số như 901 là "chín trăm lẻ một"
            return hundreds[hundred] + " lẻ " + units[rest]
        else:
            return hundreds[hundred] + " " + two_digit_number(rest)

    def group_to_vietnamese(n, add_zero_hundred):  # 9 Hàm này dùng để gọi những hàm khác theo những điều kiện dưới
        if n == 0:  # Nếu n == 0 trả về không gì cả
            return ""
        elif n < 100:  # Nếu n<100 gọi hàm two_digit_number() và với giá trị n
            return two_digit_number(n)  # 10a
        else:  # Nếu n<1000 gọi hàm three_digit_number() và với giá trị n
            if add_zero_hundred and n < 100:
                return "không trăm " + two_digit_number(n)
            return three_digit_number(n)  # 10b

    def number_to_vietnamese(n):  # 7 # Hàm tổng để chuyển số thành văn bản
        if n == 0:
            return "không"  # Nếu số là 0, trả về "không"

        list_of_number_change = []  # Danh sách lưu trữ các phần của kết quả văn bản
        unit_index = 0  # Chỉ số đơn vị lớn hiện tại (nghìn, triệu, tỉ, v.v.)
        add_zero_hundred = False

        while n > 0:  # Khi số còn lớn hơn 0
            num = n % 1000  # Lấy phần cuối cùng của số (dưới 1000)
            if num != 0:  # Điều kiện thêm để đọc số như 1018 là "một nghìn không trăm mười tám"
                part_str = group_to_vietnamese(num, add_zero_hundred)  # 8 # Chuyển đổi phần này thành văn bản tiếng Việt
                if large_units[unit_index] != "":  # Nếu đơn vị lớn hiện tại không phải là chuỗi rỗng
                    part_str += " " + large_units[unit_index]  # Thêm đơn vị lớn vào phần văn bản
                list_of_number_change.insert(0, part_str.replace("  ", " "))  # Chèn phần vào đầu danh sách `parts`, loại bỏ khoảng trắng dư thừa
            n = n // 1000  # Cập nhật số bằng cách chia cho 1000
            unit_index += 1  # Tăng chỉ số đơn vị lớn để tìm trong large_units
            add_zero_hundred = (unit_index > 1 and num == 0)

        return " ".join(list_of_number_change).strip()  # Kết hợp các phần trong danh sách `parts` thành một chuỗi văn bản và loại bỏ khoảng trắng thừa

    # Hàm thay thế số trong văn bản bằng từ
    def replace_numbers_with_words(text):  # 3
        def replace(match):  # 5
            number_str = match.group(0)  # group(0) dùng để lấy toàn bộ dữ liệu các số thuộc dạng str trong văn bản
            number = int(number_str)  # đặt number thành int của chuỗi các số đang thuộc dạng str
            return number_to_vietnamese(number)  # 6
        return re.sub(r'\d+', replace, text)  # 4 # Tìm tất cả các số và thay thế chúng

    return replace_numbers_with_words(textnumchange)  # 2


def replace_rest(text):
    text = (
        text
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "").replace("'", "")
        .replace("(", "").replace(")", "")
        .replace("{", "").replace("}", "")
        .replace("[", "").replace("]", "")
        .replace("#", "thăng")
        .replace("*", "sao")
        .replace("/", "gạch chéo")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
        .replace("=", " bằng ").replace("-", " trừ ").replace("+", " cộng ")
        .replace("_", " gạch dưới ")
        .replace("^", " mũ ")
        .replace("%", " phần trăm ")
        .replace("@", " a cồng ").replace("@gmail.com", " a cồng gờ meo chấm com ")
        .replace(" b ", " bê ").replace(" B ", " bê ")
        .replace(" c ", " sê ").replace(" C ", " sê ")
        .replace(" k ", " ca ").replace(" K ", " ca ")
        .replace(" l ", " e lờ ").replace(" L ", " e lờ ")
    )
    return text

def sum_text(a):
    text = changenumvi(a)
    return replace_rest(text)

if __name__ == "__main__":
    a = input("Nhập văn bản: ")
    print(sum_text(a))
