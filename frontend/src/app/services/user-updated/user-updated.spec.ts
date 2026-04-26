import { TestBed } from '@angular/core/testing';

import { UserUpdatedService } from './user-updated.service';

describe('UserUpdated', () => {
  let service: UserUpdatedService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(UserUpdatedService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
