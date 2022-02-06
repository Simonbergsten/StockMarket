import gc_functions.config as config

def hello_world(request):
    return "Hello, World!"


def trade_stock(request):
    data = request.get_json()

    if 'passphrase' not in data or data['passphrase'] != config.PASSPHRASE:
        return {
            "code":"error",
            "message":"You are unauthorized"
        }

    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

    try:
        order = api.submit_order(data['symbol'], data['quantity'], data['side'],
                             data['order_type'], data['time_in_force'])
    except Exception as e:
        return {
            "code": "error",
            "message": str(e)
        }

    return {
        "code": "success",
        "order_id": order.id,
        "status": order.status
    }


