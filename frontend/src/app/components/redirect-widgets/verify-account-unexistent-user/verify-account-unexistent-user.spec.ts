import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountUnexistentUser } from './verify-account-unexistent-user';

describe('VerifyAccountUnexistentUser', () => {
  let component: VerifyAccountUnexistentUser;
  let fixture: ComponentFixture<VerifyAccountUnexistentUser>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountUnexistentUser],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountUnexistentUser);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
