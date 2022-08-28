
#include "lob.h"

// outbound broadcast events. Simplified from OUCH protocol
typedef enum {EXECUTION, LIMIT_ADD, LIMIT_DEL, ORDER_ADD, ORDER_CANCEL, ORDER_REPLACE} EVENT_TYPE;

// types of fills: Filled sell, filled buy, unfilled sell, unfilled buy, modified sell, modified buy, canceled sell, canceled buy

typedef struct FilledOrder {
    
}

typedef struct ExecutionResults {
    userId_t offererId;
    orderId_t offererOrderId;
    uint64_t qty;
    vector<userId_t> offerees;
    Side side;
};